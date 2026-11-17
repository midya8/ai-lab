"""Infinite-width kernels for the ReLU MLP of net.py (NNGP and NTK recursions)."""
from __future__ import annotations

import numpy as np

from .net import SW2_HIDDEN, SW2_OUT


def _relu_moments(k12, d1, d2):
    """E[relu(u) relu(v)] and E[relu'(u) relu'(v)] for (u, v) ~ N(0, [[d1, k12], [k12, d2]])."""
    norm = np.sqrt(np.outer(d1, d2))
    cos = np.clip(k12 / np.maximum(norm, 1e-300), -1.0, 1.0)
    th = np.arccos(cos)
    e_relu = norm * (np.sin(th) + (np.pi - th) * cos) / (2 * np.pi)
    e_drelu = (np.pi - th) / (2 * np.pi)
    return e_relu, e_drelu


def _diag_next(d, sw2, sb2):
    return sw2 * d / 2.0 + sb2   # E[relu(u)^2] = d/2


def kernels(X1, X2=None, depth: int = 2, sb2: float = 0.1):
    """Return (nngp, ntk) for `depth` hidden layers.  Both are (n1, n2)."""
    X1 = np.atleast_2d(np.asarray(X1, dtype=float))
    X2 = X1 if X2 is None else np.atleast_2d(np.asarray(X2, dtype=float))
    d_in = X1.shape[1]
    sw2_first = SW2_HIDDEN if depth >= 1 else SW2_OUT
    sig = sw2_first / d_in * (X1 @ X2.T) + sb2
    d1 = sw2_first / d_in * (X1 * X1).sum(1) + sb2
    d2 = sw2_first / d_in * (X2 * X2).sum(1) + sb2
    theta = sig.copy()
    for k in range(1, depth + 1):
        sw2 = SW2_HIDDEN if k < depth else SW2_OUT
        e_relu, e_drelu = _relu_moments(sig, d1, d2)
        sig_dot = sw2 * e_drelu
        sig = sw2 * e_relu + sb2
        theta = theta * sig_dot + sig
        d1, d2 = _diag_next(d1, sw2, sb2), _diag_next(d2, sw2, sb2)
    return sig, theta


def ntk(X1, X2=None, depth: int = 2, sb2: float = 0.1) -> np.ndarray:
    return kernels(X1, X2, depth, sb2)[1]


def nngp(X1, X2=None, depth: int = 2, sb2: float = 0.1) -> np.ndarray:
    return kernels(X1, X2, depth, sb2)[0]
