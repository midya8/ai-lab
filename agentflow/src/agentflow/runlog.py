"""Persistent run log: an append-only JSONL journal of task results, with resume.

Each finished task is appended as one JSON line. A task's *fingerprint* hashes
its id, agent, prompt template and the fingerprints of its dependencies, so
editing a prompt invalidates that task and everything downstream while leaving
unrelated branches resumable. ``Orchestrator.run(tasks, log=RunLog(path),
resume=True)`` replays tasks whose latest record is ``ok`` with a matching
fingerprint instead of re-running them (no LLM call, no cost) and re-executes
only what failed, was skipped, or changed.

A crash mid-write leaves at most one truncated trailing line; readers skip
malformed lines, so the journal stays usable.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from typing import Dict, Iterable, List, Optional


def fingerprints(tasks: Iterable) -> Dict[str, str]:
    """Stable per-task hashes that chain through dependencies."""
    by_id = {t.id: t for t in tasks}
    memo: Dict[str, str] = {}

    def fp(tid: str, stack=()) -> str:
        if tid in memo:
            return memo[tid]
        if tid in stack:
            raise ValueError(f"cycle at {tid}")
        t = by_id[tid]
        payload = json.dumps(
            [t.id, t.agent, t.prompt, [fp(d, stack + (tid,)) for d in sorted(t.deps)]],
            ensure_ascii=False,
        )
        memo[tid] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return memo[tid]

    for tid in by_id:
        fp(tid)
    return memo


class RunLog:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:8]

    def record(self, run_id: str, task_id: str, fingerprint: str, result, resumed: bool = False) -> dict:
        entry = {
            "run": run_id, "ts": round(time.time(), 3), "task": task_id, "fp": fingerprint,
            "status": result.status, "output": result.output, "error": result.error,
            "attempts": result.attempts, "duration": round(result.duration, 6), "resumed": resumed,
        }
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
        return entry

    def entries(self) -> List[dict]:
        out: List[dict] = []
        try:
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue  # truncated / corrupt line
                    if isinstance(e, dict) and "task" in e and "status" in e:
                        out.append(e)
        except FileNotFoundError:
            pass
        return out

    def completed(self, tasks: Iterable) -> Dict[str, dict]:
        """Latest ``ok`` record per task whose fingerprint still matches.
        Replayed (resumed) records are ignored so provenance points at the
        run that actually produced the output."""
        fps = fingerprints(list(tasks))
        best: Dict[str, dict] = {}
        for e in self.entries():
            tid = e["task"]
            if tid in fps and e.get("fp") == fps[tid] and e["status"] == "ok" and not e.get("resumed"):
                best[tid] = e  # later lines win
        return best

    def runs(self) -> List[dict]:
        """One summary per run id, in first-seen order."""
        summ: Dict[str, dict] = {}
        for e in self.entries():
            s = summ.setdefault(e["run"], {"run": e["run"], "start": e["ts"], "tasks": 0, "ok": 0,
                                           "failed": 0, "skipped": 0, "resumed": 0, "work": 0.0})
            s["tasks"] += 1
            s[e["status"]] = s.get(e["status"], 0) + 1
            s["resumed"] += bool(e.get("resumed"))
            s["work"] = round(s["work"] + (0.0 if e.get("resumed") else e.get("duration", 0.0)), 6)
        return list(summ.values())

    def format_runs(self) -> str:
        rows = self.runs()
        if not rows:
            return "(empty log)"
        lines = [f"{'run':<10}{'tasks':>6}{'ok':>5}{'fail':>6}{'skip':>6}{'resumed':>9}{'work(s)':>9}"]
        for r in rows:
            lines.append(f"{r['run']:<10}{r['tasks']:>6}{r['ok']:>5}{r['failed']:>6}{r['skipped']:>6}"
                         f"{r['resumed']:>9}{r['work']:>9.3f}")
        return "\n".join(lines)

    def compact(self) -> int:
        """Rewrite the journal keeping only the latest non-resumed record per (task, fingerprint).
        Returns the number of lines dropped. Atomic via rename."""
        import os
        es = self.entries()
        keep: Dict[tuple, dict] = {}
        for e in es:
            if not e.get("resumed"):
                keep[(e["task"], e.get("fp"))] = e
        kept = sorted(keep.values(), key=lambda e: e["ts"])
        tmp = self.path + ".tmp"
        with self._lock:
            with open(tmp, "w", encoding="utf-8") as f:
                for e in kept:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
            os.replace(tmp, self.path)
        return len(es) - len(kept)
