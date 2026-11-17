"""Sample-quality metrics for low-dimensional generative models."""
from __future__ import annotations

import numpy as np


def sliced_wasserstein(a: np.ndarray, b: np.ndarray, n_proj: int = 128, seed: int = 0) -> float:
    """Mean 1-D Wasserstein-1 distance over random unit projections.
    Samples are subsampled to a common size (the smaller of the two)."""
    rng = np.random.default_rng(seed)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    dirs = rng.standard_normal((n_proj, a.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pa, pb = np.sort(a @ dirs.T, axis=0), np.sort(b @ dirs.T, axis=0)
    return float(np.abs(pa - pb).mean())


def mmd_rbf(a: np.ndarray, b: np.ndarray, bandwidths=(0.25, 0.5, 1.0)) -> float:
    """Unbiased squared MMD with a sum of RBF kernels."""
    def k(x, y):
        d2 = ((x[:, None, :] - y[None]) ** 2).sum(-1)
        return sum(np.exp(-d2 / (2 * h * h)) for h in bandwidths)
    n, m = len(a), len(b)
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1))
                 + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())
