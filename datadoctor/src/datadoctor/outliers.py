"""k-NN distance outliers, optionally scored within each class."""

from __future__ import annotations

import numpy as np


def knn_distance_scores(X: np.ndarray, k: int = 10, chunk: int = 512) -> np.ndarray:
    """Mean distance to the k nearest other rows (standardised features)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    k = max(1, min(k, n - 1))
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    out = np.empty(n)
    sq = (Z * Z).sum(1)
    for s in range(0, n, chunk):
        blk = Z[s : s + chunk]
        D = np.maximum(sq[s : s + chunk, None] + sq[None, :] - 2.0 * blk @ Z.T, 0.0)
        D[np.arange(D.shape[0]), np.arange(s, s + D.shape[0])] = np.inf
        part = np.partition(D, k - 1, axis=1)[:, :k]
        out[s : s + chunk] = np.sqrt(part).mean(axis=1)
    return out


def robust_z(scores: np.ndarray) -> np.ndarray:
    """(score - median) / (1.4826 * MAD): a z-score that outliers themselves cannot distort."""
    med = np.median(scores)
    mad = np.median(np.abs(scores - med))
    scale = 1.4826 * mad if mad > 1e-12 else (scores.std() or 1.0)
    return (scores - med) / scale


def find_outliers(X: np.ndarray, y: np.ndarray | None = None, k: int = 10, z: float = 3.5, min_class: int = 20):
    """Return (mask, scores). With ``y``, each class is scored against its own members, so a
    point that is normal globally but far from its labelled class is still found (classes
    smaller than ``min_class`` are scored globally)."""
    X = np.asarray(X, dtype=float)
    zs = np.zeros(len(X))
    if y is None:
        zs = robust_z(knn_distance_scores(X, k))
    else:
        y = np.asarray(y)
        rest = np.ones(len(X), dtype=bool)
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)
            if len(idx) >= min_class:
                zs[idx] = robust_z(knn_distance_scores(X[idx], k))
                rest[idx] = False
        if rest.any():
            zs[rest] = robust_z(knn_distance_scores(X, k))[rest]
    return zs > z, zs
