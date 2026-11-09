"""Shared numeric helpers: metrics, k-means, top-k selection."""
from __future__ import annotations

import numpy as np

METRICS = ("l2", "cosine", "ip")


def as_matrix(X, dim: int | None = None) -> np.ndarray:
    """float32 2-D view of X; a single vector becomes one row."""
    A = np.asarray(X, dtype=np.float32)
    if A.ndim == 1:
        A = A[None, :]
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {A.shape}")
    if dim is not None and A.shape[1] != dim:
        raise ValueError(f"expected dimension {dim}, got {A.shape[1]}")
    if not np.all(np.isfinite(A)):
        raise ValueError("vectors must be finite (no NaN or inf)")
    return A


def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared L2 distances, clipped at zero (float32)."""
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    D = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(D, 0.0, out=D)
    return D


class Metric:
    """Smaller-is-better distances for l2 / cosine / inner product.

    * l2:     squared Euclidean distance.
    * cosine: vectors are L2-normalised on the way in; the index structures then use
              squared L2 (same ranking as cosine); ``report`` converts to 1 - cos.
    * ip:     negative dot product (so that "smaller is better" holds everywhere).
    """

    def __init__(self, name: str):
        if name not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {name!r}")
        self.name = name

    def prep(self, X) -> np.ndarray:
        A = as_matrix(X)
        if self.name == "cosine":
            n = np.linalg.norm(A, axis=1, keepdims=True)
            if np.any(n == 0):
                raise ValueError("cosine metric cannot index or query a zero vector")
            A = A / n
        return np.ascontiguousarray(A)

    def dist(self, Q: np.ndarray, X: np.ndarray) -> np.ndarray:
        if self.name == "ip":
            return -(Q @ X.T)
        return sqdist(Q, X)

    def report(self, d: np.ndarray) -> np.ndarray:
        """Internal distance -> the number a user sees."""
        if self.name == "cosine":
            return d / 2.0
        return d


def topk_smallest(D: np.ndarray, k: int):
    """Row-wise k smallest entries, sorted ascending; ties break by column index.

    Returns (idx, vals). Columns holding +inf come back as idx -1.
    """
    nq, n = D.shape
    kk = min(k, n)
    if kk == 0:
        return np.full((nq, k), -1, dtype=np.int64), np.full((nq, k), np.inf, dtype=np.float32)
    if kk < n:
        part = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    else:
        part = np.tile(np.arange(n), (nq, 1))
    pv = np.take_along_axis(D, part, axis=1)
    # lexsort: primary value, secondary column index -> deterministic ties
    order = np.lexsort((part, pv), axis=-1)
    idx = np.take_along_axis(part, order, axis=1).astype(np.int64)
    vals = np.take_along_axis(pv, order, axis=1).astype(np.float32)
    idx[~np.isfinite(vals)] = -1
    if kk < k:
        idx = np.concatenate([idx, np.full((nq, k - kk), -1, dtype=np.int64)], axis=1)
        vals = np.concatenate([vals, np.full((nq, k - kk), np.inf, dtype=np.float32)], axis=1)
    return idx, vals


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0, tol: float = 1e-5):
    """Lloyd's algorithm with k-means++ seeding.

    Empty clusters are re-seeded at the point farthest from its centre. Returns
    (centres float32 (k, d), assignment int64 (n,), inertia float).
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"need at least k={k} points to fit k centres, got {n}")
    rng = np.random.default_rng(seed)
    centres = np.empty((k, d), dtype=np.float32)
    centres[0] = X[rng.integers(n)]
    d2 = sqdist(X, centres[:1])[:, 0]
    for i in range(1, k):
        tot = float(d2.sum())
        j = int(rng.choice(n, p=d2 / tot)) if tot > 0 else int(rng.integers(n))
        centres[i] = X[j]
        d2 = np.minimum(d2, sqdist(X, centres[i:i + 1])[:, 0])
    assign = np.zeros(n, dtype=np.int64)
    prev = np.inf
    inertia = np.inf
    for _ in range(iters):
        D = sqdist(X, centres)
        assign = D.argmin(axis=1)
        mind = D[np.arange(n), assign]
        inertia = float(mind.sum())
        counts = np.bincount(assign, minlength=k)
        sums = np.zeros((k, d), dtype=np.float64)
        np.add.at(sums, assign, X)
        nonempty = counts > 0
        centres[nonempty] = (sums[nonempty] / counts[nonempty, None]).astype(np.float32)
        for c in np.flatnonzero(~nonempty):
            far = int(mind.argmax())
            centres[c] = X[far]
            mind[far] = 0.0
        if prev - inertia <= tol * max(prev, 1.0) and np.isfinite(prev):
            break
        prev = inertia
    D = sqdist(X, centres)
    assign = D.argmin(axis=1)
    inertia = float(D[np.arange(n), assign].sum())
    return centres, assign.astype(np.int64), inertia
