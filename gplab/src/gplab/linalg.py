"""Small dense linear-algebra helpers (NumPy only, no SciPy)."""
from __future__ import annotations

import numpy as np


def jitter_chol(K: np.ndarray, jitter: float = 1e-10, max_tries: int = 8) -> np.ndarray:
    """Cholesky with escalating diagonal jitter (relative to the mean diagonal)."""
    n = K.shape[0]
    scale = max(float(np.mean(np.diag(K))), 1e-12)
    j = jitter * scale
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(K + j * np.eye(n))
        except np.linalg.LinAlgError:
            j *= 10.0
    raise np.linalg.LinAlgError("matrix is not positive definite even with jitter")


def chol_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve (L L^T) X = B given the lower Cholesky factor."""
    return np.linalg.solve(L.T, np.linalg.solve(L, B))


def logdet_from_chol(L: np.ndarray) -> float:
    return 2.0 * float(np.sum(np.log(np.diag(L))))


def sqdist(X: np.ndarray, Y: np.ndarray, ls: np.ndarray) -> np.ndarray:
    """Pairwise squared distance after dividing coordinates by lengthscales."""
    Xs = X / ls
    Ys = Y / ls
    d = (Xs * Xs).sum(1)[:, None] + (Ys * Ys).sum(1)[None, :] - 2.0 * Xs @ Ys.T
    return np.maximum(d, 0.0)
