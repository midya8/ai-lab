"""Tool-call-level checkpointing: never repeat a side effect when a task is retried or resumed.

``RunLog`` resumes whole tasks. But an agent task may make several tool calls (send an email,
charge a card, create a ticket) before it fails on a later step; retrying the task would repeat
them. ``ToolJournal`` records every *successful* tool call as ``(task fingerprint, call index,
tool, args) -> observation`` in an append-only JSONL file. On a retry (or a resumed run) the
agent's calls are matched positionally: while the tool name and args equal what was journaled,
the stored observation is replayed and the tool is not executed. At the first divergence the
rest of the attempt runs live. Failed calls are never journaled, so they are retried.

Assumption (disclosed): the agent is deterministic enough to re-issue the same early calls.
If the LLM changes its mind, the journal simply stops matching and nothing wrong is replayed.
The fingerprint chains the prompt and dependencies, so editing a prompt invalidates its entries.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Callable, Dict, List, Tuple


def _canon(args) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


class ToolJournal:
    def __init__(self, path: str, ttl_s: "float | None" = None, clock: Callable[[], float] = time.time):
        self.path = path
        self.ttl_s = ttl_s  # entries older than this are ignored on lookup and dropped by compact()
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, int], dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                        self._cache[(e["fp"], int(e["seq"]))] = e
                    except (ValueError, KeyError, TypeError):
                        continue  # truncated / corrupt line
        except FileNotFoundError:
            pass

    def __len__(self) -> int:
        return len(self._cache)

    def lookup(self, fp: str, seq: int, tool: str, args) -> "dict | None":
        e = self._cache.get((fp, seq))
        if e is not None and self._expired(e):
            return None
        if e is not None and e["tool"] == tool and e["args"] == json.loads(_canon(args)):
            return e
        return None

    def store(self, fp: str, seq: int, tool: str, args, obs) -> None:
        e = {"fp": fp, "seq": seq, "tool": tool, "args": json.loads(_canon(args)), "obs": str(obs), "ts": self._clock()}
        with self._lock:
            self._cache[(fp, seq)] = e
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def _expired(self, e: dict) -> bool:
        # entries without a timestamp (older journals) never expire
        return self.ttl_s is not None and "ts" in e and self._clock() - e["ts"] > self.ttl_s

    def compact(self, live_fps: "set | None" = None) -> int:
        """Rewrite the file keeping only the newest entry per (fp, seq), dropping expired entries
        and, if ``live_fps`` is given, entries of fingerprints not in it. Atomic via rename.
        Returns the number of lines removed."""
        with self._lock:
            try:
                with open(self.path, encoding="utf-8") as f:
                    before = sum(1 for line in f if line.strip())
            except FileNotFoundError:
                return 0
            keep = {k: e for k, e in self._cache.items()
                    if not self._expired(e) and (live_fps is None or k[0] in live_fps)}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for e in keep.values():
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
            self._cache = keep
            return before - len(keep)

    def scope(self, fp: str) -> "ToolScope":
        return ToolScope(self, fp)


class ToolScope:
    """One attempt's view: a call counter plus replay/live accounting."""

    def __init__(self, journal: ToolJournal, fp: str):
        self.journal, self.fp = journal, fp
        self.seq = 0
        self.replayed = 0
        self.live = 0
        self._diverged = False

    def call_sync(self, tool: str, args, fn: Callable):
        seq, self.seq = self.seq, self.seq + 1
        if not self._diverged:
            hit = self.journal.lookup(self.fp, seq, tool, args)
            if hit is not None:
                self.replayed += 1
                return hit["obs"]
            self._diverged = True
        obs = fn(tool, args)
        self.live += 1
        self.journal.store(self.fp, seq, tool, args, obs)
        return obs
