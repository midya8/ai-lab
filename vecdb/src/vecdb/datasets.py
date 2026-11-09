"""Synthetic vector sets with known structure (no downloads)."""
from __future__ import annotations

import numpy as np


def clustered(n: int, dim: int, n_clusters: int = 50, spread: float = 0.35, seed: int = 0):
    """Gaussian blobs with anisotropic spread: a stand-in for embedding clouds, where nearest
    neighbours are mostly within-cluster. Returns (vectors float32, cluster id int64)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    scales = (spread * rng.uniform(0.5, 1.5, size=(n_clusters, dim))).astype(np.float32)
    lab = rng.integers(n_clusters, size=n)
    X = centres[lab] + rng.normal(size=(n, dim)).astype(np.float32) * scales[lab]
    return X.astype(np.float32), lab.astype(np.int64)


def gaussian(n: int, dim: int, seed: int = 0) -> np.ndarray:
    """Isotropic Gaussian: no structure to exploit, the hard case for every ANN index."""
    return np.random.default_rng(seed).normal(size=(n, dim)).astype(np.float32)


def queries_like(X: np.ndarray, n: int, noise: float = 0.1, seed: int = 1) -> np.ndarray:
    """Queries = random data points plus noise (so the exact match is not trivially rank 1)."""
    rng = np.random.default_rng(seed)
    base = X[rng.choice(len(X), n, replace=False)]
    return (base + rng.normal(size=base.shape).astype(np.float32) * noise * X.std()).astype(np.float32)


def recall_at_k(found: np.ndarray, truth: np.ndarray) -> float:
    """Mean fraction of the true top-k rows that the approximate search returned."""
    k = truth.shape[1]
    hits = 0
    for f, t in zip(found, truth):
        hits += len(set(f[f >= 0].tolist()) & set(t[t >= 0].tolist()))
    return hits / (k * len(truth))
