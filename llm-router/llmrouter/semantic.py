"""Semantic cache: near-duplicate prompts hit the same entry.

Embeddings are dependency-free hashed bag-of-words + character-trigram
vectors compared by cosine similarity. A hit needs the same ``max_tokens``
and similarity >= ``threshold``. Exact normalised matches always hit.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from typing import Any

from .clock import SystemClock

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an the is are was were be to of in on for and or please can could you me my i it this that".split())


def embed(text: str, dim: int = 256) -> dict[int, float]:
    """Sparse L2-normalised hashed vector (words weigh 2x trigrams)."""
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
    vec: dict[int, float] = {}

    def add(token: str, weight: float) -> None:
        h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
        vec[h % dim] = vec.get(h % dim, 0.0) + weight

    for w in words:
        add("w:" + w, 2.0)
        padded = f"  {w} "
        for i in range(len(padded) - 2):
            add("c:" + padded[i:i + 3], 1.0)
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {k: v / norm for k, v in vec.items()} if norm else {}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


class SemanticCache:
    def __init__(self, threshold: float = 0.85, max_entries: int = 1024,
                 ttl_s: float | None = 3600.0, clock=None, dim: int = 256) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self.threshold, self.max_entries, self.ttl_s, self.dim = threshold, max_entries, ttl_s, dim
        self._clock = clock or SystemClock()
        self._data: OrderedDict[int, tuple[float, int, dict[int, float], Any]] = OrderedDict()
        self._next = 0
        self.hits = self.misses = self.semantic_hits = 0

    def _expired(self, stored_at: float) -> bool:
        return self.ttl_s is not None and self._clock.now() - stored_at >= self.ttl_s

    def lookup(self, prompt: str, max_tokens: int = 0) -> tuple[Any, float] | None:
        """Return (value, similarity) of the best live entry, or None."""
        q = embed(prompt, self.dim)
        for k in [k for k, (t, *_r) in self._data.items() if self._expired(t)]:
            del self._data[k]
        best_k, best_s = None, 0.0
        for k, (_t, mt, vec, _v) in self._data.items():
            if mt != max_tokens:
                continue
            s = cosine(q, vec)
            if s > best_s:
                best_k, best_s = k, s
        if best_k is None or best_s < self.threshold:
            self.misses += 1
            return None
        self._data.move_to_end(best_k)
        self.hits += 1
        if best_s < 0.999:
            self.semantic_hits += 1
        return self._data[best_k][3], best_s

    def store(self, prompt: str, max_tokens: int, value: Any) -> None:
        self._data[self._next] = (self._clock.now(), max_tokens, embed(prompt, self.dim), value)
        self._next += 1
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
