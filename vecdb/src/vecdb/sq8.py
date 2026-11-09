"""Scalar-quantised flat index: one uint8 per dimension (4x smaller than float32).

Per-dimension affine codes ``x ~ lo + code * step`` are fitted on the first batch (or the
data passed to ``train``); later values outside the fitted range are clipped. Search decodes
chunks on the fly (asymmetric: the query stays float32) and optionally re-ranks the top
``refine * k`` candidates exactly when ``keep_raw=True``.
"""
from __future__ import annotations

import numpy as np

from .util import Metric, as_matrix, topk_smallest

_CHUNK_ROWS = 65536


class SQ8Index:
    kind = "sq8"

    def __init__(self, dim: int, metric: str = "l2", keep_raw: bool = False):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.metric = Metric(metric)
        self.keep_raw = keep_raw
        self.lo: np.ndarray | None = None
        self.step: np.ndarray | None = None
        self._codes = np.empty((0, dim), dtype=np.uint8)
        self._raw = np.empty((0, dim), dtype=np.float32) if keep_raw else None
        self.deleted = np.zeros(0, dtype=bool)
        self.last_stats: dict = {}

    @property
    def ntotal(self) -> int:
        return self._codes.shape[0]

    @property
    def n_live(self) -> int:
        return int((~self.deleted).sum())

    @property
    def is_trained(self) -> bool:
        return self.lo is not None

    def train(self, X) -> None:
        A = self.metric.prep(as_matrix(X, self.dim))
        if len(A) == 0:
            raise ValueError("cannot train on an empty set")
        lo, hi = A.min(axis=0), A.max(axis=0)
        span = hi - lo
        # constant dimensions get step 1 so decode returns lo exactly
        self.lo = lo.astype(np.float32)
        self.step = np.where(span > 0, span / 255.0, 1.0).astype(np.float32)

    def encode(self, A: np.ndarray) -> np.ndarray:
        return np.clip(np.rint((A - self.lo) / self.step), 0, 255).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return codes.astype(np.float32) * self.step + self.lo

    def add(self, X) -> np.ndarray:
        A = self.metric.prep(as_matrix(X, self.dim))
        if not self.is_trained:
            self.train(A)
        start = self.ntotal
        self._codes = np.concatenate([self._codes, self.encode(A)])
        if self._raw is not None:
            self._raw = np.concatenate([self._raw, A])
        self.deleted = np.concatenate([self.deleted, np.zeros(len(A), dtype=bool)])
        return np.arange(start, start + len(A), dtype=np.int64)

    def remove(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.ntotal):
            raise IndexError("row out of range")
        self.deleted[rows] = True

    def reconstruct(self, rows) -> np.ndarray:
        return self.decode(self._codes[np.asarray(rows, dtype=np.int64)])

    def memory_bytes(self) -> int:
        n = int(self._codes.nbytes)
        if self._raw is not None:
            n += int(self._raw.nbytes)
        return n

    def max_abs_error(self) -> float:
        """Worst-case per-coordinate rounding error inside the fitted range."""
        return 0.0 if self.step is None else float(self.step.max() / 2)

    def search(self, Q, k: int = 10, mask=None, refine: int = 0):
        if k < 1:
            raise ValueError("k must be >= 1")
        if refine and self._raw is None:
            raise ValueError("refine needs keep_raw=True")
        Qm = self.metric.prep(as_matrix(Q, self.dim))
        ok = ~self.deleted
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.ntotal,):
                raise ValueError(f"mask must have shape ({self.ntotal},), got {m.shape}")
            ok = ok & m
        kk = min(max(k, refine * k), max(self.ntotal, 1)) if refine else k
        D = np.full((len(Qm), self.ntotal), np.inf, dtype=np.float32)
        for s in range(0, self.ntotal, _CHUNK_ROWS):
            D[:, s:s + _CHUNK_ROWS] = self.metric.dist(Qm, self.decode(self._codes[s:s + _CHUNK_ROWS]))
        D[:, ~ok] = np.inf
        ids, dists = topk_smallest(D, kk)
        if refine:
            ids2 = np.full((len(Qm), k), -1, dtype=np.int64)
            d2 = np.full((len(Qm), k), np.inf, dtype=np.float32)
            for qi in range(len(Qm)):
                cand = ids[qi][ids[qi] >= 0]
                if len(cand) == 0:
                    continue
                dd = self.metric.dist(Qm[qi:qi + 1], self._raw[cand])[0]
                o = np.argsort(dd, kind="stable")[:k]
                ids2[qi, :len(o)] = cand[o]
                d2[qi, :len(o)] = dd[o]
            ids, dists = ids2, d2
        self.last_stats = {"scanned": float(ok.sum()), "refined": float(refine * k)}
        return ids, self.metric.report(dists)

    # -- persistence ---------------------------------------------------
    def _state(self):
        arrays = {"codes": self._codes, "deleted": self.deleted}
        if self.lo is not None:
            arrays["lo"] = self.lo
            arrays["step"] = self.step
        if self._raw is not None:
            arrays["raw"] = self._raw
        return arrays, {"dim": self.dim, "metric": self.metric.name, "keep_raw": self.keep_raw}

    @classmethod
    def _from_state(cls, arrays, meta):
        idx = cls(meta["dim"], meta["metric"], keep_raw=meta["keep_raw"])
        idx._codes = np.ascontiguousarray(arrays["codes"], dtype=np.uint8)
        idx.deleted = arrays["deleted"].astype(bool)
        if "lo" in arrays:
            idx.lo = arrays["lo"].astype(np.float32)
            idx.step = arrays["step"].astype(np.float32)
        if meta["keep_raw"]:
            idx._raw = np.ascontiguousarray(arrays["raw"], dtype=np.float32)
        return idx
