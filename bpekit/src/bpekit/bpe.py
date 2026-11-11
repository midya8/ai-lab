"""Byte-level BPE tokenizer trained from scratch (GPT-2 style, no regex pre-split lib)."""
from __future__ import annotations
import json, re
from collections import Counter

_SPLIT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


def _merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new); i += 2
        else:
            out.append(ids[i]); i += 1
    return out


def train(text: str, vocab_size: int = 300, specials=("<|endoftext|>",)) -> "BPETokenizer":
    if vocab_size < 256:
        raise ValueError("vocab_size must be >= 256")
    words = Counter(_SPLIT.findall(text))
    seqs = {w: list(w.encode("utf-8")) for w in words}
    merges = []
    for n in range(256, vocab_size):
        stats = Counter()
        for w, s in seqs.items():
            for p in zip(s, s[1:]):
                stats[p] += words[w]
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))  # deterministic ties
        if stats[best] < 2:
            break
        merges.append(best)
        for w in seqs:
            seqs[w] = _merge(seqs[w], best, n)
    return BPETokenizer(merges, specials)


class BPETokenizer:
    def __init__(self, merges, specials=()):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        base = 256 + len(self.merges)
        self.specials = {s: base + i for i, s in enumerate(specials)}
        self._inv_special = {v: k for k, v in self.specials.items()}
        self._cache = {}

    def __len__(self):
        return 256 + len(self.merges) + len(self.specials)

    def _encode_word(self, w):
        if w in self._cache:
            return self._cache[w]
        ids = list(w.encode("utf-8"))
        while len(ids) > 1:
            pair = min(zip(ids, ids[1:]), key=lambda p: self.rank.get(p, float("inf")))
            if pair not in self.rank:
                break
            ids = _merge(ids, pair, 256 + self.rank[pair])
        self._cache[w] = ids
        return ids

    def encode(self, text: str):
        if not self.specials:
            parts = [text]
        else:
            pat = "(" + "|".join(re.escape(s) for s in self.specials) + ")"
            parts = re.split(pat, text)
        out = []
        for part in parts:
            if part in self.specials:
                out.append(self.specials[part])
            else:
                for w in _SPLIT.findall(part):
                    out.extend(self._encode_word(w))
        return out

    def decode(self, ids) -> str:
        b = b"".join(self._inv_special[i].encode() if i in self._inv_special else self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def compression(self, text: str) -> float:
        """bytes per token."""
        return len(text.encode("utf-8")) / max(1, len(self.encode(text)))

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges, "specials": list(self.specials)}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(d["merges"], d["specials"])
