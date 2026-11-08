"""Thread-safe LRU + TTL answer cache keyed on a normalised question.

Normalisation is deliberately conservative: case, punctuation and runs of
whitespace are ignored, but word order and every word are kept. That means
"How long do I have to return a bike?" and "how long do i have to return a bike"
share an entry, while rephrasings do not (they could legitimately score
differently, and a wrong cache hit is worse than a miss).
"""
from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from typing import Callable

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def normalize(question: str) -> str:
    return " ".join(_NON_WORD.sub(" ", question.casefold()).split())


class AnswerCache:
    def __init__(self, max_size: int = 256, ttl: float | None = 300.0,
                 clock: Callable[[], float] = time.monotonic):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0 (or None for no expiry)")
        self.max_size = max_size
        self.ttl = ttl
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = self.evictions = self.expirations = 0

    def get(self, question: str):
        """Return the cached value or None; a hit refreshes LRU position."""
        key = normalize(question)
        with self._lock:
            item = self._data.get(key)
            if item is not None:
                stored_at, value = item
                if self.ttl is not None and self._clock() - stored_at >= self.ttl:
                    del self._data[key]
                    self.expirations += 1
                else:
                    self._data.move_to_end(key)
                    self.hits += 1
                    return value
            self.misses += 1
            return None

    def put(self, question: str, value) -> None:
        key = normalize(question)
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._data), "max_size": self.max_size, "ttl": self.ttl,
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "expirations": self.expirations,
                "hit_rate": round(self.hits / total, 4) if total else 0.0,
            }
