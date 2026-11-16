"""Sparse GP regression with inducing inputs: DTC, FITC and the Titsias (2009) VFE bound.

All three share one collapsed form. With ``A = L_uu^-1 K_uf`` and a per-point noise
``Lambda`` the approximate log marginal likelihood is

    log N(y | 0, Q_ff + Lambda) - correction,    Q_ff = A^T A

* DTC:   Lambda = sigma^2 I,                           correction = 0
* FITC:  Lambda = diag(K_ff - Q_ff) + sigma^2 I,       correction = 0
* VFE:   Lambda = sigma^2 I,   correction = tr(K_ff - Q_ff) / (2 sigma^2)   (a true lower bound)

Cost is O(n m^2) time and O(n m) memory.
"""
from __future__ import annotations

import numpy as np

from .gp import LOG2PI
from .kernels import Kernel, _as2d
from .linalg import jitter_chol
from .optim import lbfgs

METHODS = ("vfe", "fitc", "dtc")


def greedy_inducing(kernel: Kernel, X, m: int, seed: int = 0) -> np.ndarray:
    """Indices of ``m`` inducing points by greedy pivoted-Cholesky on K_ff: each step adds
    the point with the largest remaining prior variance, i.e. the one that most reduces
    ``tr(K_ff - Q_ff)``. Needs O(n m) kernel entries, never the full matrix."""
    X = _as2d(X)
    n = X.shape[0]
    m = min(m, n)
    d = kernel.diag(X).astype(float).copy()
    idx: list[int] = []
    L = np.zeros((m, n))
    rng = np.random.default_rng(seed)
    for j in range(m):
        # break exact ties randomly so constant-variance kernels do not always take index 0
        cand = np.flatnonzero(d >= d.max() - 1e-12 * max(d.max(), 1.0))
        p = int(rng.choice(cand)) if cand.size > 1 and j == 0 else int(cand[0])
        if d[p] <= 1e-12:
            break
        idx.append(p)
        col = kernel(X, X[p:p + 1])[:, 0] - L[:j].T @ L[:j, p]
        L[j] = col / np.sqrt(d[p])
        d = np.maximum(d - L[j] ** 2, 0.0)
        d[p] = 0.0
    return np.array(idx, dtype=int)


def kmeans_inducing(X, m: int, seed: int = 0, iters: int = 25) -> np.ndarray:
    """Lloyd's k-means centres (not data points) as inducing inputs."""
    X = _as2d(X)
    rng = np.random.default_rng(seed)
    C = X[rng.choice(X.shape[0], size=min(m, X.shape[0]), replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(C.shape[0]):
            if np.any(a == j):
                C[j] = X[a == j].mean(0)
    return C


class SparseGP:
    """Sparse GP with fixed inducing inputs ``Z`` and a choice of objective."""

    def __init__(self, kernel: Kernel, Z, noise: float = 0.1, method: str = "vfe",
                 center: bool = False, jitter: float = 1e-8):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        self.kernel = kernel
        self.Z = _as2d(Z).copy()
        self.log_noise = float(np.log(noise))
        self.method = method
        self.center = center
        self.jitter = jitter
        self.X = None

    @property
    def noise(self) -> float:
        return float(np.exp(self.log_noise))

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.kernel.params, [self.log_noise]])

    @params.setter
    def params(self, theta):
        theta = np.asarray(theta, float)
        self.kernel.params = theta[:-1]
        self.log_noise = float(theta[-1])
        self._factor()

    def set_data(self, X, y):
        self.X = _as2d(X)
        y = np.asarray(y, float).ravel()
        self.mean_ = float(y.mean()) if self.center else 0.0
        self.y = y - self.mean_
        self._factor()
        return self

    def _factor(self):
        if self.X is None:
            return
        n, m = self.X.shape[0], self.Z.shape[0]
        Kuu = self.kernel(self.Z)
        self._Luu = jitter_chol(Kuu, self.jitter)
        Kuf = self.kernel(self.Z, self.X)
        self._A = np.linalg.solve(self._Luu, Kuf)                    # m x n
        qdiag = np.sum(self._A * self._A, axis=0)
        self._resid = np.maximum(self.kernel.diag(self.X) - qdiag, 0.0)   # diag(K_ff - Q_ff)
        s2 = self.noise
        self._lam = s2 + self._resid if self.method == "fitc" else np.full(n, s2)
        At = self._A / np.sqrt(self._lam)
        yt = self.y / np.sqrt(self._lam)
        B = np.eye(m) + At @ At.T
        self._LB = np.linalg.cholesky(B)
        self._LBinv = np.linalg.inv(self._LB)
        self._c = self._LBinv @ (At @ yt)
        self._yt = yt

    def bound(self) -> float:
        """The collapsed objective (a lower bound on the exact lml for ``vfe``)."""
        n = self.y.size
        quad = float(self._yt @ self._yt - self._c @ self._c)
        logdet = float(np.sum(np.log(self._lam))) + 2.0 * float(np.sum(np.log(np.diag(self._LB))))
        val = -0.5 * (n * LOG2PI + logdet + quad)
        if self.method == "vfe":
            val -= float(self._resid.sum()) / (2.0 * self.noise)
        return val

    def trace_gap(self) -> float:
        """tr(K_ff - Q_ff): the quantity VFE penalises and greedy selection shrinks."""
        return float(self._resid.sum())

    def fit(self, X, y, max_iter: int = 60, eps: float = 1e-5) -> "SparseGP":
        """Maximise the objective over kernel and noise parameters (central finite-difference
        gradients: there are only a handful of parameters) with Z held fixed."""
        self.set_data(X, y)

        def fun(theta):
            self.params = theta
            f = -self.bound()
            g = np.zeros_like(theta)
            for i in range(theta.size):
                e = np.zeros_like(theta)
                e[i] = eps
                self.params = theta + e
                fp = -self.bound()
                self.params = theta - e
                fm = -self.bound()
                g[i] = (fp - fm) / (2 * eps)
            self.params = theta
            return f, g

        th, _, _ = lbfgs(fun, self.params, max_iter=max_iter)
        self.params = th
        return self

    def predict(self, Xs, include_noise: bool = False):
        Xs = _as2d(Xs)
        W = np.linalg.solve(self._Luu, self.kernel(self.Z, Xs))      # m x n*
        V = self._LBinv @ W
        mean = V.T @ self._c + self.mean_
        var = self.kernel.diag(Xs) - np.sum(W * W, axis=0) + np.sum(V * V, axis=0)
        var = np.maximum(var, 1e-12)
        if include_noise:
            var = var + self.noise
        return mean, var
