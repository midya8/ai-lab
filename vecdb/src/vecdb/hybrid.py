"""Hybrid lexical + vector search: BM25 and k-NN fused with reciprocal-rank or weighted-score fusion.

``HybridCollection`` wraps a :class:`Collection`. Each id may carry a text; BM25 is computed over
those texts (stdlib + numpy, built lazily and rebuilt when the collection changes). Metadata filters
apply to both retrievers, so a hit never violates ``where``.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import numpy as np

from .collection import Collection, Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of documents (``k1`` saturation, ``b`` length normalisation)."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.len = np.array([len(d) for d in docs], dtype=np.float64)
        self.avg = float(self.len.mean()) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post.setdefault(t, []).append((i, c))

    def idf(self, term: str) -> float:
        df = len(self.post.get(term, ()))
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(self.n, dtype=np.float64)
        for t in set(query):
            idf = self.idf(t)
            for i, tf in self.post.get(t, ()):
                norm = self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1.0))
                s[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        return s


class HybridCollection:
    def __init__(self, dim: int, metric: str = "cosine", index: str = "flat", **index_params):
        self.col = Collection(dim, metric, index, **index_params)
        self._text: dict[str, str] = {}
        self._bm25: tuple[int, BM25, list[str]] | None = None

    def __len__(self) -> int:
        return len(self.col)

    def upsert(self, ids, vectors, metadatas=None, texts=None) -> None:
        ids_l = [ids] if isinstance(ids, str) else list(ids)
        if texts is not None:
            texts = [texts] if isinstance(texts, str) else list(texts)
            if len(texts) != len(ids_l):
                raise ValueError(f"{len(ids_l)} ids but {len(texts)} texts")
        self.col.upsert(ids_l, vectors, metadatas)  # validates before we touch text
        for j, i in enumerate(ids_l):
            if texts is not None:
                self._text[i] = texts[j]
            else:
                self._text.pop(i, None)
        self._bm25 = None

    def delete(self, ids) -> int:
        ids_l = [ids] if isinstance(ids, str) else list(ids)
        for i in ids_l:
            self._text.pop(i, None)
        self._bm25 = None
        return self.col.delete(ids_l)

    def _lex(self) -> tuple[BM25, list[str]]:
        if self._bm25 is None or self._bm25[0] != self.col._version:
            ids = [i for i in self._text if i in self.col]
            self._bm25 = (self.col._version, BM25([tokenize(self._text[i]) for i in ids]), ids)
        return self._bm25[1], self._bm25[2]

    def lexical(self, text: str, k: int = 10, where: dict | None = None) -> list[tuple[str, float]]:
        """Top-k (id, BM25 score) among live, filter-passing texts; zero-score docs are omitted."""
        bm, ids = self._lex()
        if not ids:
            return []
        sc = bm.scores(tokenize(text))
        if where:
            ok = self.col._mask(where)
            keep = np.array([ok[self.col._row_of[i]] for i in ids], dtype=bool)
            sc = np.where(keep, sc, 0.0)
        order = np.argsort(-sc, kind="stable")[:k]
        return [(ids[j], float(sc[j])) for j in order if sc[j] > 0]

    def query(self, vector, text: str, k: int = 10, where: dict | None = None, fusion: str = "rrf",
              alpha: float = 0.5, rrf_k: int = 60, depth: int | None = None, **search_params) -> list[Hit]:
        """Fuse vector and BM25 rankings.

        ``fusion='rrf'``: sum of 1/(rrf_k + rank) over both lists (scale-free, no tuning).
        ``fusion='weighted'``: alpha * minmax(vector similarity) + (1-alpha) * minmax(BM25).
        ``distance`` on returned hits is the negated fused score (smaller is better, as elsewhere).
        """
        if fusion not in ("rrf", "weighted"):
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        depth = depth or max(4 * k, 50)
        dense = self.col.query(np.asarray(vector, dtype=np.float32)[None], depth, where, **search_params)[0]
        lex = self.lexical(text, depth, where)
        fused: dict[str, float] = {}
        if fusion == "rrf":
            for r, h in enumerate(dense):
                fused[h.id] = fused.get(h.id, 0.0) + 1.0 / (rrf_k + r + 1)
            for r, (i, _) in enumerate(lex):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r + 1)
        else:
            def norm(vals):
                lo, hi = min(vals), max(vals)
                return [1.0 if hi == lo else (v - lo) / (hi - lo) for v in vals]
            if dense:
                for h, v in zip(dense, norm([-h.distance for h in dense])):
                    fused[h.id] = fused.get(h.id, 0.0) + alpha * v
            if lex:
                for (i, _), v in zip(lex, norm([s for _, s in lex])):
                    fused[i] = fused.get(i, 0.0) + (1 - alpha) * v
        top = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [Hit(i, -s, dict(self.col._meta[self.col._row_of[i]])) for i, s in top]

    # -- persistence ----------------------------------------------------
    def save(self, directory: str) -> None:
        """Save the vector collection plus texts (``texts.json``); BM25 is rebuilt lazily on load."""
        self.col.save(directory)
        tmp = os.path.join(directory, "texts.json.tmp")
        with open(tmp, "w") as f:
            json.dump({i: t for i, t in self._text.items() if i in self.col}, f)
        os.replace(tmp, os.path.join(directory, "texts.json"))

    @classmethod
    def load(cls, directory: str) -> "HybridCollection":
        col = Collection.load(directory)
        h = cls.__new__(cls)
        h.col = col
        h._bm25 = None
        path = os.path.join(directory, "texts.json")
        h._text = {}
        if os.path.exists(path):
            with open(path) as f:
                h._text = json.load(f)
        unknown = [i for i in h._text if i not in col]
        if unknown:
            raise ValueError(f"texts.json has ids not in the collection: {unknown[:3]}")
        return h
