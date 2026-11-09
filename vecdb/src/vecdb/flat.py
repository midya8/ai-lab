"""Exact brute-force index: the ground truth every approximate index is scored against."""
from __future__ import annotations

import numpy as np

from .util import Metric, as_matrix, topk_smallest

_CHUNK_BYTES = 128 * 1024 * 1024


class FlatIndex:
    kind = "flat"

    def __init__(self, dim: int, metric: str = "l2"):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.metric = Metric(metric)
        self._vecs = np.empty((0, dim), dtype=np.float32)
        self.deleted = np.zeros(0, dtype=bool)
        self.last_stats: dict = {}

    # -- bookkeeping ---------------------------------------------------
    @property
    def ntotal(self) -> int:
        return self._vecs.shape[0]

    @property
    def n_live(self) -> int:
        return int((~self.deleted).sum())

    def add(self, X) -> np.ndarray:
        A = self.metric.prep(as_matrix(X, self.dim))
        start = self.ntotal
        self._vecs = np.concatenate([self._vecs, A], axis=0)
        self.deleted = np.concatenate([self.deleted, np.zeros(len(A), dtype=bool)])
        return np.arange(start, start + len(A), dtype=np.int64)

    def remove(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.ntotal):
            raise IndexError("row out of range")
        self.deleted[rows] = True

    def reconstruct(self, rows) -> np.ndarray:
        return self._vecs[np.asarray(rows, dtype=np.int64)].copy()

    def memory_bytes(self) -> int:
        return int(self._vecs.nbytes)

    def _allowed(self, mask) -> np.ndarray | None:
        if mask is None and not self.deleted.any():
            return None
        ok = ~self.deleted
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.ntotal,):
                raise ValueError(f"mask must have shape ({self.ntotal},), got {m.shape}")
            ok = ok & m
        return ok

    # -- search --------------------------------------------------------
    def search(self, Q, k: int = 10, mask=None):
        """Exact k nearest rows. Returns (rows, distances); -1 / inf pad missing slots."""
        if k < 1:
            raise ValueError("k must be >= 1")
        Qm = self.metric.prep(as_matrix(Q, self.dim))
        ok = self._allowed(mask)
        n = max(self.ntotal, 1)
        step = max(1, _CHUNK_BYTES // (4 * n))
        ids, dists = [], []
        for s in range(0, len(Qm), step):
            D = self.metric.dist(Qm[s:s + step], self._vecs)
            if ok is not None:
                D[:, ~ok] = np.inf
            i, v = topk_smallest(D, k)
            ids.append(i)
            dists.append(v)
        self.last_stats = {"scanned": float(self.n_live if mask is None else (ok.sum() if ok is not None else self.ntotal))}
        return np.concatenate(ids), self.metric.report(np.concatenate(dists))

    # -- persistence ---------------------------------------------------
    def _state(self):
        return {"vecs": self._vecs, "deleted": self.deleted}, {"dim": self.dim, "metric": self.metric.name}

    @classmethod
    def _from_state(cls, arrays, meta):
        idx = cls(meta["dim"], meta["metric"])
        idx._vecs = np.ascontiguousarray(arrays["vecs"], dtype=np.float32)
        idx.deleted = arrays["deleted"].astype(bool)
        return idx
