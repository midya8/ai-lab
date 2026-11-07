"""Byte-level BPE tokenizer trained from scratch (GPT-2 style, no regex pre-split).

Drop-in compatible with CharTokenizer: ``vocab_size``, ``encode``, ``decode``.
Base vocabulary is the 256 bytes, so any string round-trips losslessly.
"""
from __future__ import annotations

import json
from collections import Counter


def _merge(ids: list[int], pair: tuple[int, int], new: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self._build()

    def _build(self) -> None:
        self.rank = {p: i for i, p in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

    @classmethod
    def train(cls, text: str, vocab_size: int) -> "BPETokenizer":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        ids = list(text.encode("utf-8"))
        merges: list[tuple[int, int]] = []
        for k in range(vocab_size - 256):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            best, n = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if n < 2:
                break
            merges.append(best)
            ids = _merge(ids, best, 256 + k)
        return cls(merges)

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges)

    def encode(self, s: str) -> list[int]:
        ids = list(s.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            best = min(pairs, key=lambda p: self.rank.get(p, float("inf")))
            if best not in self.rank:
                break
            ids = _merge(ids, best, 256 + self.rank[best])
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            return cls([tuple(m) for m in json.load(f)["merges"]])
