"""A tiny vector database: string ids, metadata, filtered k-NN, upserts, deletes, persistence."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .filters import compile_filter
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .io import load_index, save_index
from .ivf import IVFIndex
from .util import as_matrix


@dataclass
class Hit:
    id: str
    distance: float
    metadata: dict = field(default_factory=dict)


class Collection:
    """Rows live in an index (``flat``, ``hnsw`` or ``ivf``); the collection adds an id <-> row map,
    JSON metadata per row, metadata filtering that is pushed *into* the search, and upsert/delete.

    Upsert of an existing id tombstones its old row and appends a new one, so ``stats()['dead_rows']``
    grows until you ``compact()`` (rebuild from live vectors; unavailable for IVF-PQ without
    ``keep_raw`` because the original vectors are gone).
    """

    def __init__(self, dim: int, metric: str = "cosine", index: str = "hnsw", **index_params):
        self.dim, self.metric_name, self.index_kind = dim, metric, index
        self.index_params = dict(index_params)
        self.index = self._new_index()
        self._ids: list[str] = []          # row -> external id
        self._meta: list[dict] = []        # row -> metadata
        self._row_of: dict[str, int] = {}  # live id -> row
        self._version = 0
        self._mask_cache: dict[str, tuple[int, np.ndarray]] = {}

    def _new_index(self):
        p = self.index_params
        if self.index_kind == "flat":
            return FlatIndex(self.dim, self.metric_name, **p)
        if self.index_kind == "hnsw":
            return HNSWIndex(self.dim, self.metric_name, **p)
        if self.index_kind == "ivf":
            return IVFIndex(self.dim, metric=self.metric_name, **p)
        raise ValueError("index must be 'flat', 'hnsw' or 'ivf'")

    def __len__(self) -> int:
        return len(self._row_of)

    def __contains__(self, id_: str) -> bool:
        return id_ in self._row_of

    # -- writes ---------------------------------------------------------
    def upsert(self, ids, vectors, metadatas=None) -> None:
        ids = [ids] if isinstance(ids, str) else list(ids)
        V = as_matrix(vectors, self.dim)
        if len(ids) != len(V):
            raise ValueError(f"{len(ids)} ids but {len(V)} vectors")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids within one upsert call")
        if not all(isinstance(i, str) and i for i in ids):
            raise ValueError("ids must be non-empty strings")
        metas = [{} for _ in ids] if metadatas is None else [dict(m) for m in metadatas]
        if len(metas) != len(ids):
            raise ValueError(f"{len(ids)} ids but {len(metas)} metadata dicts")
        for m in metas:
            try:
                json.dumps(m)  # must be serialisable; fail before mutating anything
            except (TypeError, ValueError) as e:
                raise ValueError(f"metadata must be JSON-serialisable: {e}") from None
        self.index.metric.prep(V)  # validate (e.g. zero vectors under cosine) before mutating
        old = [self._row_of.pop(i) for i in ids if i in self._row_of]
        if old:
            self.index.remove(old)
        rows = self.index.add(V)
        for r, i, m in zip(rows, ids, metas):
            assert r == len(self._ids)
            self._ids.append(i)
            self._meta.append(m)
            self._row_of[i] = int(r)
        self._version += 1

    def delete(self, ids) -> int:
        ids = [ids] if isinstance(ids, str) else list(ids)
        rows = [self._row_of.pop(i) for i in ids if i in self._row_of]
        if rows:
            self.index.remove(rows)
            self._version += 1
        return len(rows)

    def train(self, X=None) -> None:
        """IVF only: fit the coarse quantiser on ``X`` or on everything upserted so far."""
        if self.index_kind != "ivf":
            raise ValueError("train() applies to ivf collections only")
        self.index.train(X)

    # -- reads ----------------------------------------------------------
    def get(self, id_: str):
        r = self._row_of.get(id_)
        if r is None:
            raise KeyError(id_)
        return self.index.reconstruct([r])[0], dict(self._meta[r])

    def _mask(self, where: dict) -> np.ndarray:
        key = json.dumps(where, sort_keys=True, default=str)
        hit = self._mask_cache.get(key)
        if hit is not None and hit[0] == self._version:
            return hit[1]
        pred = compile_filter(where)
        mask = np.zeros(len(self._ids), dtype=bool)
        for i in self._row_of.values():
            mask[i] = pred(self._meta[i])
        if len(self._mask_cache) > 64:
            self._mask_cache.clear()
        self._mask_cache[key] = (self._version, mask)
        return mask

    def query(self, vectors, k: int = 10, where: dict | None = None, **search_params):
        """k nearest live rows that satisfy ``where``. One list of :class:`Hit` per query vector.

        ``search_params`` go to the index (``ef`` for hnsw; ``nprobe``/``refine`` for ivf).
        """
        mask = self._mask(where) if where else None
        ids, dist = self.index.search(vectors, k, mask=mask, **search_params)
        out = []
        for ri, di in zip(ids, dist):
            out.append([Hit(self._ids[r], float(d), dict(self._meta[r])) for r, d in zip(ri, di) if r >= 0])
        return out

    def query_mmr(self, vector, k: int = 10, fetch_k: int = 50, lambda_mult: float = 0.5,
                  where: dict | None = None, **search_params):
        """Maximal Marginal Relevance: fetch ``fetch_k`` candidates, then greedily pick ``k`` that balance
        relevance to the query (``lambda_mult``) against similarity to already-picked hits (cosine).
        ``lambda_mult=1`` equals plain top-k; ``0`` is maximum diversity. Single query vector only."""
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError("lambda_mult must be in [0, 1]")
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        cands = self.query(q[None, :], k=max(fetch_k, k), where=where, **search_params)[0]
        if not cands:
            return []
        V = np.stack([self.get(h.id)[0] for h in cands]).astype(np.float64)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        qn = q / max(float(np.linalg.norm(q)), 1e-12)
        rel = V @ qn
        chosen: list[int] = []
        rest = list(range(len(cands)))
        while rest and len(chosen) < k:
            if chosen:
                red = (V[rest] @ V[chosen].T).max(axis=1)
            else:
                red = np.zeros(len(rest))
            score = lambda_mult * rel[rest] - (1 - lambda_mult) * red
            best = rest[int(np.argmax(score))]
            chosen.append(best)
            rest.remove(best)
        return [cands[i] for i in chosen]

    def count(self, where: dict | None = None) -> int:
        if not where:
            return len(self)
        return int(self._mask(where).sum())

    def stats(self) -> dict:
        return {"live": len(self), "dead_rows": self.index.ntotal - len(self), "index": self.index_kind,
                "metric": self.metric_name, "bytes": self.index.memory_bytes()}

    def compact(self) -> None:
        """Rebuild the index from live rows only, reclaiming tombstoned space."""
        live = sorted(self._row_of.values())
        if self.index_kind == "ivf" and self.index.pq is not None and not self.index.keep_raw:
            raise ValueError("compact() would need the original vectors; build the IVF-PQ index with keep_raw=True")
        V = self.index.reconstruct(live) if live else np.empty((0, self.dim), dtype=np.float32)
        ids = [self._ids[r] for r in live]
        metas = [self._meta[r] for r in live]
        was_trained = self.index_kind != "ivf" or self.index.trained
        self.index = self._new_index()
        self._ids, self._meta, self._row_of = [], [], {}
        if self.index_kind == "ivf" and was_trained:
            self.index.train(V)
        if len(V):
            self.upsert(ids, V, metas)
        self._version += 1

    # -- persistence ----------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        save_index(self.index, os.path.join(directory, "index.npz"))
        doc = {"dim": self.dim, "metric": self.metric_name, "index": self.index_kind,
               "index_params": self.index_params, "ids": self._ids, "meta": self._meta,
               "live": sorted(self._row_of.values())}
        tmp = os.path.join(directory, "collection.json.tmp")
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, os.path.join(directory, "collection.json"))

    @classmethod
    def load(cls, directory: str) -> "Collection":
        with open(os.path.join(directory, "collection.json")) as f:
            doc = json.load(f)
        c = cls(doc["dim"], doc["metric"], doc["index"], **doc["index_params"])
        c.index = load_index(os.path.join(directory, "index.npz"))
        c._ids, c._meta = doc["ids"], doc["meta"]
        c._row_of = {c._ids[r]: r for r in doc["live"]}
        if len(c._ids) != c.index.ntotal:
            raise ValueError("collection.json and index.npz disagree on row count")
        return c
