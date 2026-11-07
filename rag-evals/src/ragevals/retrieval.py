"""Okapi BM25 retriever implemented from scratch."""
from __future__ import annotations
import math
import re
from collections import Counter

from .chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("a an and are as at be by for from in is it of on or that the to was were with what which who how why when".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = (sum(self.len) / len(chunks)) if chunks else 0.0
        df: Counter = Counter()
        for t in self.tf:
            df.update(t.keys())
        n = len(chunks)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, ln in zip(self.tf, self.len):
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * ln / (self.avg or 1))
                    s += self.idf[w] * f * (self.k1 + 1) / denom
            out.append(s)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
        return [(self.chunks[i], scores[i]) for i in order if scores[i] > 0]
