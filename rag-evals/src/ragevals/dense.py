"""Dense retrieval via TF-IDF + truncated SVD (latent semantic analysis) and hybrid RRF fusion."""
from __future__ import annotations
import math
from collections import Counter

import numpy as np

from .chunking import Chunk
from .retrieval import BM25Index, tokenize


class LSAIndex:
    """TF-IDF vectors projected into a low-rank latent space with SVD."""

    def __init__(self, chunks: list[Chunk], dim: int = 32):
        self.chunks = chunks
        docs = [tokenize(c.text) for c in chunks]
        vocab = sorted({w for d in docs for w in d})
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n = max(len(chunks), 1)
        df = Counter(w for d in docs for w in set(d))
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in vocab]) if vocab else np.zeros(0)
        X = np.stack([self._tfidf(d) for d in docs]) if docs and vocab else np.zeros((len(chunks), 0))
        self.dim = 0
        if X.size:
            U, S, Vt = np.linalg.svd(X, full_matrices=False)
            self.dim = min(dim, len(S))
            self.Vt = Vt[: self.dim]
            self.emb = self._norm(X @ self.Vt.T)
        else:
            self.emb = np.zeros((len(chunks), 0))

    def _tfidf(self, toks: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(toks).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        return v

    @staticmethod
    def _norm(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M / np.where(n == 0, 1, n)

    def score(self, query: str) -> list[float]:
        if not self.dim:
            return [0.0] * len(self.chunks)
        q = self._norm(self._tfidf(tokenize(query)) @ self.Vt.T)
        return (self.emb @ q).tolist()

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        s = self.score(query)
        order = sorted(range(len(s)), key=lambda i: (-s[i], i))[:k]
        return [(self.chunks[i], s[i]) for i in order if s[i] > 1e-9]


def rrf_fuse(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_r w_r / (k + rank_r(d)), ranks 1-based."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda t: (-t[1], t[0]))


class HybridIndex:
    """BM25 + LSA fused with RRF."""

    def __init__(self, chunks: list[Chunk], dim: int = 32, rrf_k: int = 60, weights=(1.0, 1.0)):
        self.chunks = chunks
        self.bm25 = BM25Index(chunks)
        self.lsa = LSAIndex(chunks, dim)
        self.rrf_k, self.weights = rrf_k, list(weights)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        pos = {c.id: i for i, c in enumerate(self.chunks)}
        rankings = [[pos[c.id] for c, _ in idx.search(query, len(self.chunks))] for idx in (self.bm25, self.lsa)]
        fused = rrf_fuse(rankings, self.rrf_k, self.weights)[:k]
        return [(self.chunks[i], s) for i, s in fused]
