"""Inverted-file index (IVF-Flat) with optional product-quantised residuals (IVF-PQ)."""
from __future__ import annotations

import numpy as np

from .pq import ProductQuantizer
from .util import Metric, as_matrix, kmeans, topk_smallest


class IVFIndex:
    """Coarse k-means partitions the space into ``nlist`` cells; a query scans only the
    ``nprobe`` closest cells.

    * ``pq_m=None``: IVF-Flat, cells hold the raw vectors (exact distances inside a cell).
    * ``pq_m=m``: IVF-PQ, cells hold ``m``-byte codes of the *residual* ``x - centroid`` and the
      query is scored by ADC table lookups. ``keep_raw=True`` additionally keeps the raw
      vectors so ``search(..., refine=r)`` can re-rank the top ``r*k`` ADC hits exactly.

    Vectors added before :meth:`train` are buffered and indexed when training happens.
    """

    kind = "ivf"

    def __init__(self, dim: int, nlist: int = 64, metric: str = "l2", pq_m: int | None = None,
                 pq_bits: int = 8, keep_raw: bool = False, seed: int = 0, max_train: int | None = None):
        if nlist < 1:
            raise ValueError("nlist must be >= 1")
        self.dim, self.nlist, self.seed = dim, nlist, seed
        self.metric = Metric(metric)
        if pq_m is not None and self.metric.name == "ip":
            raise ValueError("residual PQ needs an L2-compatible metric: use 'l2' or 'cosine', not 'ip'")
        self.pq = ProductQuantizer(dim, pq_m, pq_bits, seed) if pq_m else None
        self.keep_raw = bool(keep_raw and self.pq is not None)
        self.max_train = max_train if max_train is not None else 256 * nlist
        self.centroids: np.ndarray | None = None
        self._ids: list[np.ndarray] = []
        self._data: list[np.ndarray] = []
        self._pending = np.empty((0, dim), dtype=np.float32)
        self._raw = np.empty((0, dim), dtype=np.float32)
        self.ntotal = 0
        self.deleted = np.zeros(0, dtype=bool)
        self.last_stats: dict = {}

    @property
    def trained(self) -> bool:
        return self.centroids is not None

    @property
    def n_live(self) -> int:
        return int((~self.deleted).sum())

    # -- training / adding --------------------------------------------
    def train(self, X=None) -> "IVFIndex":
        if X is None:
            if len(self._pending) == 0:
                raise ValueError("train() needs data: pass X or add() vectors first")
            T = self._pending
        else:
            T = self.metric.prep(as_matrix(X, self.dim))
        if len(T) < self.nlist:
            raise ValueError(f"need at least nlist={self.nlist} training vectors, got {len(T)}")
        if len(T) > self.max_train:
            sel = np.random.default_rng(self.seed).choice(len(T), self.max_train, replace=False)
            T = T[np.sort(sel)]
        self.centroids, _, _ = kmeans(T, self.nlist, seed=self.seed)
        if self.pq is not None:
            a = self.metric.dist(T, self.centroids).argmin(axis=1)
            self.pq.fit(T - self.centroids[a])
        self._ids = [np.empty(0, dtype=np.int64) for _ in range(self.nlist)]
        width = self.pq.m if self.pq is not None else self.dim
        dtype = np.uint8 if self.pq is not None else np.float32
        self._data = [np.empty((0, width), dtype=dtype) for _ in range(self.nlist)]
        if len(self._pending):
            pend, self._pending = self._pending, np.empty((0, self.dim), dtype=np.float32)
            self._store(pend, np.arange(len(pend), dtype=np.int64))
        return self

    def add(self, X) -> np.ndarray:
        A = self.metric.prep(as_matrix(X, self.dim))
        rows = np.arange(self.ntotal, self.ntotal + len(A), dtype=np.int64)
        self.ntotal += len(A)
        self.deleted = np.concatenate([self.deleted, np.zeros(len(A), dtype=bool)])
        if self.trained:
            self._store(A, rows)
        else:
            self._pending = np.concatenate([self._pending, A])
        return rows

    def _store(self, A: np.ndarray, rows: np.ndarray) -> None:
        a = self.metric.dist(A, self.centroids).argmin(axis=1)
        if self.pq is not None:
            codes = self.pq.encode(A - self.centroids[a])
            payload = codes
        else:
            payload = A
        if self.keep_raw:
            self._raw = np.concatenate([self._raw, A])
        order = np.argsort(a, kind="stable")
        a_sorted = a[order]
        bounds = np.searchsorted(a_sorted, np.arange(self.nlist + 1))
        for l in range(self.nlist):
            lo, hi = bounds[l], bounds[l + 1]
            if hi > lo:
                sel = order[lo:hi]
                self._ids[l] = np.concatenate([self._ids[l], rows[sel]])
                self._data[l] = np.concatenate([self._data[l], payload[sel]])

    def remove(self, rows) -> None:
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= self.ntotal):
            raise IndexError("row out of range")
        self.deleted[rows] = True

    # -- search ---------------------------------------------------------
    def search(self, Q, k: int = 10, nprobe: int = 8, mask=None, refine: int | None = None):
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self.trained:
            raise RuntimeError("IVFIndex is not trained; call train() first")
        if refine is not None and not self.keep_raw:
            raise ValueError("refine needs keep_raw=True on an IVF-PQ index")
        nprobe = max(1, min(int(nprobe), self.nlist))
        Qm = self.metric.prep(as_matrix(Q, self.dim))
        ok = ~self.deleted
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != (self.ntotal,):
                raise ValueError(f"mask must have shape ({self.ntotal},), got {m.shape}")
            ok = ok & m
        out_i = np.full((len(Qm), k), -1, dtype=np.int64)
        out_d = np.full((len(Qm), k), np.inf, dtype=np.float32)
        scanned = 0
        cd_all = self.metric.dist(Qm, self.centroids)
        for qi, q in enumerate(Qm):
            cd = cd_all[qi]
            probes = np.argpartition(cd, nprobe - 1)[:nprobe] if nprobe < self.nlist else np.arange(self.nlist)
            ids_parts, d_parts = [], []
            for l in probes:
                ids = self._ids[l]
                if len(ids) == 0:
                    continue
                if self.pq is None:
                    d = self.metric.dist(q[None, :], self._data[l])[0]
                else:
                    table = self.pq.adc_table(q - self.centroids[l])
                    d = self.pq.adc_distances(table, self._data[l])
                ids_parts.append(ids)
                d_parts.append(d)
            if not ids_parts:
                continue
            ids = np.concatenate(ids_parts)
            d = np.concatenate(d_parts)
            scanned += len(ids)
            keep = ok[ids]
            ids, d = ids[keep], d[keep]
            if len(ids) == 0:
                continue
            if refine is not None:
                r = min(len(ids), max(k, refine * k))
                top, _ = topk_smallest(d[None, :], r)
                ids = ids[top[0]]
                d = self.metric.dist(q[None, :], self._raw[ids])[0]
            i, v = topk_smallest(d[None, :], k)
            sel = i[0] >= 0
            out_i[qi, sel] = ids[i[0][sel]]
            out_d[qi, sel] = v[0][sel]
        self.last_stats = {"scanned": scanned / max(len(Qm), 1), "nprobe": nprobe}
        return out_i, self.metric.report(out_d)

    # -- misc -----------------------------------------------------------
    def reconstruct(self, rows) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if not self.trained:
            return self._pending[rows].copy()
        if self.keep_raw:
            return self._raw[rows].copy()
        out = np.empty((len(rows), self.dim), dtype=np.float32)
        found = np.zeros(len(rows), dtype=bool)
        for l in range(self.nlist):
            hit = np.isin(rows, self._ids[l])
            if not hit.any():
                continue
            for j in np.flatnonzero(hit):
                p = int(np.flatnonzero(self._ids[l] == rows[j])[0])
                v = self._data[l][p]
                out[j] = v if self.pq is None else self.pq.decode(v[None, :])[0] + self.centroids[l]
                found[j] = True
        if not found.all():
            raise IndexError("row not found")
        return out

    def list_sizes(self) -> np.ndarray:
        return np.array([len(i) for i in self._ids], dtype=np.int64)

    def memory_bytes(self) -> int:
        n = self._pending.nbytes + self._raw.nbytes
        if self.trained:
            n += self.centroids.nbytes + sum(a.nbytes for a in self._ids) + sum(a.nbytes for a in self._data)
            if self.pq is not None:
                n += self.pq.codebooks.nbytes
        return int(n)

    def _state(self):
        arrays = {"deleted": self.deleted, "pending": self._pending, "raw": self._raw}
        meta = {"dim": self.dim, "nlist": self.nlist, "metric": self.metric.name, "seed": self.seed,
                "pq_m": self.pq.m if self.pq else None, "pq_bits": self.pq.nbits if self.pq else 8,
                "keep_raw": self.keep_raw, "max_train": self.max_train, "ntotal": self.ntotal,
                "trained": self.trained}
        if self.trained:
            arrays["centroids"] = self.centroids
            sizes = self.list_sizes()
            arrays["sizes"] = sizes
            arrays["ids"] = np.concatenate(self._ids) if sizes.sum() else np.empty(0, dtype=np.int64)
            width = self.pq.m if self.pq else self.dim
            arrays["data"] = np.concatenate(self._data) if sizes.sum() else np.empty((0, width))
            if self.pq is not None:
                arrays["codebooks"] = self.pq.codebooks
        return arrays, meta

    @classmethod
    def _from_state(cls, arrays, meta):
        idx = cls(meta["dim"], meta["nlist"], meta["metric"], meta["pq_m"], meta["pq_bits"],
                  meta["keep_raw"], meta["seed"], meta["max_train"])
        idx.ntotal = meta["ntotal"]
        idx.deleted = arrays["deleted"].astype(bool)
        idx._pending = np.ascontiguousarray(arrays["pending"], dtype=np.float32)
        idx._raw = np.ascontiguousarray(arrays["raw"], dtype=np.float32)
        if meta["trained"]:
            idx.centroids = np.ascontiguousarray(arrays["centroids"], dtype=np.float32)
            if idx.pq is not None:
                idx.pq.codebooks = np.ascontiguousarray(arrays["codebooks"], dtype=np.float32)
            sizes = arrays["sizes"]
            off = np.concatenate([[0], np.cumsum(sizes)])
            dtype = np.uint8 if idx.pq is not None else np.float32
            data = arrays["data"].astype(dtype)
            idx._ids = [arrays["ids"][off[l]:off[l + 1]].astype(np.int64) for l in range(idx.nlist)]
            idx._data = [data[off[l]:off[l + 1]] for l in range(idx.nlist)]
        return idx
