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

import contextlib
import json
import os
import threading
import time
try:
    import fcntl
except ImportError:  # non-POSIX: falls back to thread-only locking
    fcntl = None
from typing import Callable, Dict, List, Tuple


def _canon(args) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


class ToolJournal:
    def __init__(self, path: str, ttl_s: "float | None" = None, clock: Callable[[], float] = time.time,
                 max_lines: "int | None" = None):
        self.path = path
        self.max_lines = max_lines  # auto-compact when the file holds more than this many lines
        self.auto_compactions = 0
        self._lines = 0
        self.ttl_s = ttl_s  # entries older than this are ignored on lookup and dropped by compact()
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: Dict[Tuple[str, int], dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self._lines += 1
                    try:
                        e = json.loads(line)
                        self._cache[(e["fp"], int(e["seq"]))] = e
                    except (ValueError, KeyError, TypeError):
                        continue  # truncated / corrupt line
        except FileNotFoundError:
            pass

    @contextlib.contextmanager
    def _flock(self):
        """Exclusive cross-process lock on a sidecar ``<path>.lock`` file (POSIX only)."""
        if fcntl is None:
            yield
            return
        with open(self.path + ".lock", "a") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def refresh(self) -> None:
        """Re-read the file so entries written by other processes become visible."""
        with self._lock:
            self._lines = 0
            self._load()

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
        with self._lock, self._flock():
            self._cache[(fp, seq)] = e
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            self._lines += 1
            if self.max_lines is not None and self._lines > self.max_lines:
                self._lines = 0
                self._load()  # merge other processes' entries before rewriting
                self._compact_locked(None)
                self.auto_compactions += 1

    def _expired(self, e: dict) -> bool:
        # entries without a timestamp (older journals) never expire
        return self.ttl_s is not None and "ts" in e and self._clock() - e["ts"] > self.ttl_s

    def compact(self, live_fps: "set | None" = None) -> int:
        """Rewrite the file keeping only the newest entry per (fp, seq), dropping expired entries
        and, if ``live_fps`` is given, entries of fingerprints not in it. Atomic via rename.
        Returns the number of lines removed."""
        with self._lock, self._flock():
            self._lines = 0
            self._load()  # never drop entries another process appended
            return self._compact_locked(live_fps)

    def _compact_locked(self, live_fps) -> int:
        before = self._lines
        keep = {k: e for k, e in self._cache.items()
                if not self._expired(e) and (live_fps is None or k[0] in live_fps)}
        if not os.path.exists(self.path):
            return 0
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for e in keep.values():
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        self._cache = keep
        self._lines = len(keep)
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
