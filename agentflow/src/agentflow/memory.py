"""Thread-safe shared memory that agents read and write through tools.

Agents get `remember(key, value)`, `recall(key)`, `search(query)` and
`forget(key)` tools via `attach_memory`. Entries record the writing agent,
and search ranks by token overlap (key counts double). Optional JSON
persistence uses an atomic write.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from typing import Dict, List, Optional

from .core import Agent


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class SharedMemory:
    def __init__(self, path: Optional[str] = None, ttl: Optional[float] = None, clock=time.time):
        self.ttl = ttl  # default seconds-to-live for entries; None = never expire
        self._clock = clock
        self._data: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self.path = path
        if path and os.path.exists(path):
            with open(path) as f:
                self._data = json.load(f)

    def remember(self, key: str, value: str, author: str = "", ttl: Optional[float] = None) -> None:
        ttl = self.ttl if ttl is None else ttl
        entry = {"value": value, "author": author}
        if ttl is not None:
            entry["expires_at"] = self._clock() + ttl
        with self._lock:
            self._data[key] = entry
            self._save()

    def _expired(self, e: dict) -> bool:
        return "expires_at" in e and self._clock() >= e["expires_at"]

    def purge(self) -> int:
        """Drop expired entries; returns how many were removed."""
        with self._lock:
            dead = [k for k, e in self._data.items() if self._expired(e)]
            for k in dead:
                del self._data[k]
            if dead:
                self._save()
            return len(dead)

    def recall(self, key: str) -> Optional[str]:
        with self._lock:
            e = self._data.get(key)
            return e["value"] if e and not self._expired(e) else None

    def forget(self, key: str) -> bool:
        with self._lock:
            e = self._data.pop(key, None)
            found = e is not None and not self._expired(e)
            if found:
                self._save()
            return found

    def search(self, query: str, limit: int = 3) -> List[dict]:
        q = _tokens(query)
        scored = []
        with self._lock:
            for k, e in self._data.items():
                if self._expired(e):
                    continue
                score = 2 * len(q & _tokens(k)) + len(q & _tokens(e["value"]))
                if score:
                    scored.append((-score, k, {"key": k, **e}))
        scored.sort(key=lambda t: (t[0], t[1]))
        return [t[2] for t in scored[:limit]]

    def keys(self) -> List[str]:
        with self._lock:
            return sorted(k for k, e in self._data.items() if not self._expired(e))

    def _save(self) -> None:
        if not self.path:
            return
        d = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)


def attach_memory(agent: Agent, memory: SharedMemory) -> Agent:
    """Register memory tools on the agent (author is stamped automatically)."""
    t = agent.tools
    t.register("remember", lambda key, value: (memory.remember(key, value, agent.name), "stored")[1])
    t.register("recall", lambda key: memory.recall(key) if memory.recall(key) is not None else "not found")
    t.register("search", lambda query: memory.search(query))
    t.register("forget", lambda key: memory.forget(key))
    return agent
