"""Classical (pairwise) Hopfield networks on +-1 states.

Three storage rules for the symmetric, zero-diagonal weight matrix W:

* ``hebbian``       W = X^T X / N                 capacity ~0.138 N for random patterns
* ``pseudoinverse`` W = X^T (X X^T)^-1 X          projection rule, exact for linearly independent patterns
* ``storkey``       local incremental rule         higher capacity than Hebb, still local

Dynamics are asynchronous: neurons update one at a time against the field produced by the *other*
neurons, which makes the energy E(s) = -1/2 s^T W s non-increasing (a Lyapunov function), so every
run reaches a fixed point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _check_patterns(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("patterns must have shape (p, n)")
    if not np.all(np.abs(X) == 1.0):
        raise ValueError("patterns must be +-1")
    return X


def hebbian(X: np.ndarray) -> np.ndarray:
    X = _check_patterns(X)
    W = X.T @ X / X.shape[1]
    np.fill_diagonal(W, 0.0)
    return W


def pseudoinverse(X: np.ndarray, zero_diag: bool = True) -> np.ndarray:
    """Projection rule. With ``zero_diag=False`` W is exactly the orthogonal projector onto span(X)."""
    X = _check_patterns(X)
    W = X.T @ np.linalg.pinv(X @ X.T) @ X
    W = 0.5 * (W + W.T)
    if zero_diag:
        np.fill_diagonal(W, 0.0)
    return W


def storkey(X: np.ndarray) -> np.ndarray:
    """Storkey's rule (1997): W_ij += (x_i x_j - x_i h_ji - h_ij x_j) / N with h_ij = sum_{k!=i,j} W_ik x_k."""
    X = _check_patterns(X)
    n = X.shape[1]
    W = np.zeros((n, n))
    for x in X:
        field = W @ x                       # (W x)_i, W has zero diagonal so no self term
        h = field[:, None] - W * x[None, :]  # h_ij = (W x)_i - W_ij x_j   (k != i, j)
        W = W + (np.outer(x, x) - x[:, None] * h.T - h * x[None, :]) / n
        np.fill_diagonal(W, 0.0)
    return W


def energy(W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """E(s) = -1/2 s^T W s for each row of S (diagonal of W ignored)."""
    S = np.atleast_2d(S)
    d = np.diag(W)
    return -0.5 * (np.einsum("bi,ij,bj->b", S, W, S) - (S * S) @ d)


@dataclass
class Result:
    states: np.ndarray          # (b, n) final states
    sweeps: np.ndarray          # (b,) sweeps used until no change (or max_sweeps)
    converged: np.ndarray       # (b,) bool
    energies: np.ndarray        # (sweeps_run+1, b) energy after each sweep (row 0 = start)


def retrieve(W: np.ndarray, S0: np.ndarray, max_sweeps: int = 50, rng: np.random.Generator | None = None) -> Result:
    """Asynchronous retrieval for a batch of start states.

    Each sweep visits every neuron once in a random order shared across the batch; each state is still
    updated strictly sequentially, so the per-state energy never increases.
    """
    rng = rng or np.random.default_rng(0)
    S = np.atleast_2d(np.asarray(S0, dtype=float)).copy()
    b, n = S.shape
    diag = np.diag(W).copy()
    sweeps = np.zeros(b, dtype=int)
    converged = np.zeros(b, dtype=bool)
    energies = [energy(W, S)]
    for _ in range(max_sweeps):
        changed = np.zeros(b, dtype=bool)
        for i in rng.permutation(n):
            h = S @ W[i] - diag[i] * S[:, i]
            new = np.where(h > 1e-12, 1.0, np.where(h < -1e-12, -1.0, S[:, i]))
            changed |= new != S[:, i]
            S[:, i] = new
        active = ~converged
        sweeps[active] += 1
        converged |= ~changed
        energies.append(energy(W, S))
        if converged.all():
            break
    return Result(S, sweeps, converged, np.array(energies))


def is_fixed_point(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Per-pattern bool: sign(field_i) == x_i for every neuron (strictly stable)."""
    X = np.atleast_2d(X)
    H = X @ W - X * np.diag(W)[None, :]
    return np.all(H * X > 0, axis=1)
