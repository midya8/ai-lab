"""Response cache: normalised exact-match, LRU eviction, TTL expiry."""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any

from .clock import SystemClock


def cache_key(prompt: str, max_tokens: int = 0) -> str:
    """Key insensitive to case and whitespace differences."""
    norm = re.sub(r"\s+", " ", prompt.strip().lower())
    return hashlib.sha256(f"{max_tokens}|{norm}".encode()).hexdigest()


class ResponseCache:
    def __init__(self, max_entries: int = 1024, ttl_s: float | None = 3600.0,
                 clock=None) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self.max_entries = max_entries
        self.ttl_s = ttl_s
        self._clock = clock or SystemClock()
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            self.misses += 1
            return None
        stored_at, value = item
        if self.ttl_s is not None and self._clock.now() - stored_at >= self.ttl_s:
            del self._data[key]
            self.misses += 1
            return None
        self._data.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: Any) -> None:
        self._data[key] = (self._clock.now(), value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
