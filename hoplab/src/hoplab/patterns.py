"""Pattern generators and corruption helpers (all deterministic given a numpy Generator)."""
from __future__ import annotations

import numpy as np


def random_patterns(p: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """p i.i.d. uniform +-1 patterns of length n, shape (p, n)."""
    return rng.choice(np.array([-1.0, 1.0]), size=(p, n))


def correlated_patterns(p: int, n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """+-1 patterns that each agree with one shared template with probability (1+rho)/2.

    Two patterns then have expected overlap rho^2 (per unit), so rho=0 is the random case and
    rho->1 makes them nearly identical. Hebbian storage degrades quickly with rho.
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    template = rng.choice(np.array([-1.0, 1.0]), size=n)
    keep = rng.random((p, n)) < (1.0 + rho) / 2.0
    X = np.where(keep, template[None, :], -template[None, :])
    return X


def flip_noise(x: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Flip exactly round(frac*n) random entries of each row of x (+-1)."""
    x = np.atleast_2d(np.asarray(x, dtype=float)).copy()
    n = x.shape[1]
    k = int(round(frac * n))
    for r in range(x.shape[0]):
        idx = rng.choice(n, size=k, replace=False)
        x[r, idx] *= -1.0
    return x


def mask_noise(x: np.ndarray, idx: np.ndarray, value: float = -1.0) -> np.ndarray:
    """Overwrite coordinates idx of each row with a constant (occlusion)."""
    x = np.atleast_2d(np.asarray(x, dtype=float)).copy()
    x[:, idx] = value
    return x


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised overlap (1/n) a.b between rows of a and rows of b (broadcast row-wise)."""
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    return np.sum(a * b, axis=1) / a.shape[1]
