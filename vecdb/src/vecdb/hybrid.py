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
        ``alpha`` also weights RRF (dense list x 2*alpha, BM25 list x 2*(1-alpha)); the default 0.5 is plain RRF.
        ``fusion='weighted'``: alpha * minmax(vector similarity) + (1-alpha) * minmax(BM25).
        ``fusion='zscore'``: like 'weighted' but each list is standardised (mean 0, std 1) instead of
        min-max scaled, so one outlier score cannot squash the rest of its list.
        ``distance`` on returned hits is the negated fused score (smaller is better, as elsewhere).
        """
        if fusion not in ("rrf", "weighted", "zscore"):
            raise ValueError("fusion must be 'rrf', 'weighted' or 'zscore'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        depth = depth or max(4 * k, 50)
        dense = self.col.query(np.asarray(vector, dtype=np.float32)[None], depth, where, **search_params)[0]
        lex = self.lexical(text, depth, where)
        fused: dict[str, float] = {}
        if fusion == "rrf":
            for r, h in enumerate(dense):
                fused[h.id] = fused.get(h.id, 0.0) + 2 * alpha / (rrf_k + r + 1)
            for r, (i, _) in enumerate(lex):
                fused[i] = fused.get(i, 0.0) + 2 * (1 - alpha) / (rrf_k + r + 1)
        else:
            def norm(vals):
                if fusion == "zscore":
                    a = np.asarray(vals, dtype=np.float64)
                    sd = a.std()
                    # shift so the worst kept hit is 0: absent docs score 0, as in min-max
                    z = np.zeros_like(a) if sd == 0 else (a - a.mean()) / sd
                    return list(z - z.min() + (1.0 if sd == 0 else 0.0))
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

    def query_mmr(self, vector, text: str, k: int = 10, lambda_: float = 0.7, pool: int | None = None,
                  **kwargs) -> list[Hit]:
        """Maximal Marginal Relevance re-rank of the fused candidates.

        Greedily picks the hit maximising ``lambda_*rel - (1-lambda_)*max_sim_to_selected`` where ``rel`` is the
        fused score min-max scaled to [0, 1] and similarity is cosine between stored vectors. ``lambda_=1``
        reproduces ``query``; smaller values trade relevance for diversity. Candidates come from the top
        ``pool`` (default ``3*k``) fused hits.
        """
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("lambda_ must be in [0, 1]")
        cands = self.query(vector, text, pool or 3 * k, **kwargs)
        if len(cands) <= 1:
            return cands[:k]
        sc = np.array([-h.distance for h in cands], dtype=np.float64)
        rel = np.ones_like(sc) if sc.max() == sc.min() else (sc - sc.min()) / (sc.max() - sc.min())
        V = np.stack([self.col.get(h.id)[0] for h in cands]).astype(np.float64)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        sim = V @ V.T
        chosen: list[int] = []
        rest = list(range(len(cands)))
        while rest and len(chosen) < k:
            if chosen:
                red = sim[np.ix_(rest, chosen)].max(axis=1)
            else:
                red = np.zeros(len(rest))
            val = lambda_ * rel[rest] - (1 - lambda_) * red
            j = int(np.argmax(val))  # first max = better fused rank on ties
            chosen.append(rest.pop(j))
        return [cands[j] for j in chosen]

    def query_rerank(self, vector, text: str, scorer, k: int = 10, candidates: int | None = None,
                     blend: float = 1.0, **kwargs) -> list[Hit]:
        """Two-stage retrieval: fuse to ``candidates`` (default 5*k), then re-score with ``scorer``.

        ``scorer(query_text, doc_texts) -> sequence of floats`` (higher is better), e.g. a cross-encoder or
        any callable. ``blend`` in [0, 1] mixes the min-max-scaled scorer output (weight ``blend``) with the
        min-max-scaled first-stage fused score (weight ``1-blend``); 1.0 means the scorer alone decides.
        Candidates without stored text get the scorer's input ``""``. ``distance`` is the negated final score.
        """
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        pool = self.query(vector, text, k=candidates or 5 * k, **kwargs)
        if not pool:
            return []
        sc = np.asarray(list(scorer(text, [self._text.get(h.id, "") for h in pool])), dtype=np.float64)
        if sc.shape != (len(pool),):
            raise ValueError("scorer must return one score per candidate")

        def mm(a):
            lo, hi = a.min(), a.max()
            return np.ones_like(a) if hi == lo else (a - lo) / (hi - lo)
        first = mm(np.asarray([-h.distance for h in pool], dtype=np.float64))
        final = blend * mm(sc) + (1 - blend) * first
        order = sorted(range(len(pool)), key=lambda j: (-final[j], pool[j].id))[:k]
        return [Hit(pool[j].id, -float(final[j]), pool[j].metadata) for j in order]

    def query_batch(self, vectors, texts: list[str], k: int = 10, **kwargs) -> list[list[Hit]]:
        """Run ``query`` for each (vector, text) pair; results are in input order."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(texts):
            raise ValueError("vectors must be (n, dim) with one text per row")
        return [self.query(v, t, k, **kwargs) for v, t in zip(vectors, texts)]

    def explain(self, vector, text: str, k: int = 10, where: dict | None = None, depth: int | None = None,
                rrf_k: int = 60, **search_params) -> list[dict]:
        """RRF query that reports why each hit ranked where it did.

        Each row: ``id``, ``score`` (RRF), ``dense_rank`` / ``lex_rank`` (1-based, None if absent from that
        list's top ``depth``), ``dense_distance`` and ``bm25`` (None when absent), and ``source``
        ('both', 'dense' or 'lexical').
        """
        depth = depth or max(4 * k, 50)
        dense = self.col.query(np.asarray(vector, dtype=np.float32)[None], depth, where, **search_params)[0]
        lex = self.lexical(text, depth, where)
        rows: dict[str, dict] = {}
        def row(i):
            return rows.setdefault(i, {"id": i, "score": 0.0, "dense_rank": None, "lex_rank": None,
                                       "dense_distance": None, "bm25": None})
        for r, h in enumerate(dense):
            d = row(h.id)
            d["score"] += 1.0 / (rrf_k + r + 1)
            d["dense_rank"], d["dense_distance"] = r + 1, float(h.distance)
        for r, (i, sc) in enumerate(lex):
            d = row(i)
            d["score"] += 1.0 / (rrf_k + r + 1)
            d["lex_rank"], d["bm25"] = r + 1, sc
        top = sorted(rows.values(), key=lambda d: (-d["score"], d["id"]))[:k]
        for d in top:
            d["source"] = ("both" if d["dense_rank"] and d["lex_rank"]
                           else "dense" if d["dense_rank"] else "lexical")
        return top

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
