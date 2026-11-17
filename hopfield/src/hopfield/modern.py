"""Modern Hopfield networks.

Two families:

* Binary dense associative memories (Krotov & Hopfield 2016; Demircigil et al. 2017). Energy
  E(s) = -sum_mu F(m_mu), m_mu = (xi^mu . s)/N, with F rectified-polynomial or exponential. Asynchronous
  updates pick the sign that lowers E, so E is a Lyapunov function and capacity grows with the
  interaction order (polynomial) or exponentially in N (exp).

* Continuous states (Ramsauer et al. 2020). With patterns X (p, d), the update
  xi <- X^T softmax(beta X xi) is exactly one step of softmax attention with queries xi, keys and
  values X, and temperature beta; it minimises the energy
  E(xi) = -lse(beta, X xi) + 1/2 xi.xi + log(p)/beta + 1/2 M^2  (M = max ||x||) by a CCCP argument.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ------------------------------------------------------------------ binary dense associative memory


def _F(m: np.ndarray, kind: str, order: int, beta: float) -> np.ndarray:
    if kind == "poly":
        return np.maximum(m, 0.0) ** order
    if kind == "exp":
        return np.exp(beta * m)
    raise ValueError("kind must be 'poly' or 'exp'")


def dam_energy(X: np.ndarray, S: np.ndarray, kind: str = "exp", order: int = 3, beta: float = 8.0) -> np.ndarray:
    """E(s) = -sum_mu F(X s / N) for each row of S. For kind='exp' the log-sum-exp form is returned
    scaled by 1/beta (a monotone transform, same minima, numerically stable)."""
    S = np.atleast_2d(S)
    M = S @ X.T / X.shape[1]
    if kind == "exp":
        z = beta * M
        zmax = z.max(axis=1, keepdims=True)
        return -(zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))) / beta
    return -_F(M, kind, order, beta).sum(axis=1)


@dataclass
class DamResult:
    states: np.ndarray
    sweeps: np.ndarray
    converged: np.ndarray
    energies: np.ndarray   # (sweeps_run+1, b)


def dam_retrieve(X: np.ndarray, S0: np.ndarray, kind: str = "exp", order: int = 3, beta: float = 8.0,
                 max_sweeps: int = 20, rng: np.random.Generator | None = None) -> DamResult:
    """Asynchronous dense-associative-memory retrieval (batched; each state updated sequentially).

    Neuron i takes the sign minimising E: s_i <- sign(sum_mu F(m_mu | s_i=+1) - F(m_mu | s_i=-1)).
    """
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, dtype=float)
    S = np.atleast_2d(np.asarray(S0, dtype=float)).copy()
    b, n = S.shape
    A = S @ X.T                                   # (b, p) raw overlaps, updated incrementally
    sweeps = np.zeros(b, dtype=int)
    converged = np.zeros(b, dtype=bool)
    energies = [dam_energy(X, S, kind, order, beta)]
    for _ in range(max_sweeps):
        changed = np.zeros(b, dtype=bool)
        for i in rng.permutation(n):
            xi = X[:, i][None, :]                 # (1, p)
            rest = A - xi * S[:, i:i + 1]         # overlap with neuron i removed
            mp = (rest + xi) / n
            mm = (rest - xi) / n
            if kind == "exp":
                c = np.maximum(mp.max(axis=1), mm.max(axis=1))[:, None]
                field = (np.exp(beta * (mp - c)) - np.exp(beta * (mm - c))).sum(axis=1)
            else:
                field = (_F(mp, kind, order, beta) - _F(mm, kind, order, beta)).sum(axis=1)
            new = np.where(field > 1e-300, 1.0, np.where(field < -1e-300, -1.0, S[:, i]))
            delta = new - S[:, i]
            A += delta[:, None] * xi
            changed |= delta != 0
            S[:, i] = new
        active = ~converged
        sweeps[active] += 1
        converged |= ~changed
        energies.append(dam_energy(X, S, kind, order, beta))
        if converged.all():
            break
    return DamResult(S, sweeps, converged, np.array(energies))


# ------------------------------------------------------------------ continuous modern Hopfield / attention


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def cont_update(X: np.ndarray, Xi: np.ndarray, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """One retrieval step for a batch of query states Xi (b, d): returns (new states, attention weights)."""
    P = softmax(beta * np.atleast_2d(Xi) @ X.T, axis=-1)
    return P @ X, P


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Plain scaled dot-product attention softmax(Q K^T * scale) V (scale defaults to 1/sqrt(d_k))."""
    scale = 1.0 / np.sqrt(K.shape[-1]) if scale is None else scale
    return softmax(Q @ K.T * scale, axis=-1) @ V


def cont_energy(X: np.ndarray, Xi: np.ndarray, beta: float) -> np.ndarray:
    Xi = np.atleast_2d(Xi)
    z = beta * Xi @ X.T
    zmax = z.max(axis=1, keepdims=True)
    lse = (zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))) / beta
    M = np.max(np.linalg.norm(X, axis=1))
    return -lse + 0.5 * np.sum(Xi * Xi, axis=1) + np.log(X.shape[0]) / beta + 0.5 * M * M


@dataclass
class ContResult:
    states: np.ndarray
    weights: np.ndarray       # final attention weights (b, p)
    steps: np.ndarray         # steps until ||delta|| < tol
    energies: np.ndarray      # (steps_run+1, b)


def cont_retrieve(X: np.ndarray, Xi0: np.ndarray, beta: float, max_steps: int = 100, tol: float = 1e-9) -> ContResult:
    Xi = np.atleast_2d(np.asarray(Xi0, dtype=float)).copy()
    b = Xi.shape[0]
    steps = np.zeros(b, dtype=int)
    done = np.zeros(b, dtype=bool)
    energies = [cont_energy(X, Xi, beta)]
    W = None
    for _ in range(max_steps):
        new, W = cont_update(X, Xi, beta)
        move = np.linalg.norm(new - Xi, axis=1)
        steps[~done] += 1
        done |= move < tol
        Xi = new
        energies.append(cont_energy(X, Xi, beta))
        if done.all():
            break
    if W is None:
        W = cont_update(X, Xi, beta)[1]
    return ContResult(Xi, W, steps, np.array(energies))


def effective_number(P: np.ndarray) -> np.ndarray:
    """exp(entropy) of attention weights: 1 for a one-hot (retrieval), p for uniform (global average)."""
    P = np.atleast_2d(P)
    H = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1.0)), 0.0), axis=1)
    return np.exp(H)


def separation(X: np.ndarray) -> np.ndarray:
    """Delta_i = x_i.x_i - max_{j!=i} x_i.x_j: how far each stored pattern stands from its nearest rival."""
    G = X @ X.T
    d = np.diag(G).copy()
    np.fill_diagonal(G, -np.inf)
    return d - G.max(axis=1)
