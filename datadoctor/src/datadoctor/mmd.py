"""Multivariate drift: unbiased RBF-kernel MMD^2 with a permutation null.

Per-feature tests (KS/PSI) miss shifts that live only in the joint
distribution, e.g. a changed correlation with unchanged marginals. MMD sees
them. Bandwidth defaults to the median heuristic on the pooled sample.
"""
from __future__ import annotations

import numpy as np


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2 * a @ b.T, 0.0)


def median_bandwidth(Z: np.ndarray) -> float:
    d = _sqdist(Z, Z)[np.triu_indices(len(Z), 1)]
    m = float(np.median(d))
    return float(np.sqrt(m / 2)) if m > 0 else 1.0


def _mmd2_from_kernel(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    n, m = len(idx_a), len(idx_b)
    Kaa, Kbb, Kab = K[np.ix_(idx_a, idx_a)], K[np.ix_(idx_b, idx_b)], K[np.ix_(idx_a, idx_b)]
    return float((Kaa.sum() - np.trace(Kaa)) / (n * (n - 1))
                 + (Kbb.sum() - np.trace(Kbb)) / (m * (m - 1)) - 2 * Kab.mean())


def mmd_drift(X_train, X_test, n_perm: int = 200, bandwidth: float | None = None,
              standardize: bool = True, seed: int = 0, alpha: float = 0.01,
              max_n: int = 1000) -> dict:
    """Unbiased MMD^2 between two samples and a permutation p-value.

    Subsamples each side to max_n rows. p = (1 + #{perm >= observed}) / (1 + n_perm).
    """
    A, B = np.asarray(X_train, float), np.asarray(X_test, float)
    if A.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("train and test must be 2-D with the same number of features")
    if len(A) < 2 or len(B) < 2:
        raise ValueError("need at least 2 rows per sample")
    rng = np.random.default_rng(seed)
    if len(A) > max_n:
        A = A[rng.choice(len(A), max_n, replace=False)]
    if len(B) > max_n:
        B = B[rng.choice(len(B), max_n, replace=False)]
    Z = np.vstack([A, B])
    if standardize:
        Z = (Z - A.mean(0)) / np.where(A.std(0) > 0, A.std(0), 1.0)
    bw = bandwidth or median_bandwidth(Z)
    K = np.exp(-_sqdist(Z, Z) / (2 * bw**2))
    n = len(A)
    ia, ib = np.arange(n), np.arange(n, len(Z))
    obs = _mmd2_from_kernel(K, ia, ib)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        if _mmd2_from_kernel(K, p[:n], p[n:]) >= obs:
            count += 1
    pval = (1 + count) / (1 + n_perm)
    return {"mmd2": obs, "p_value": pval, "bandwidth": bw, "drifted": bool(pval < alpha)}
