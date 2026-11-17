"""Intrinsic coregionalisation model (ICM) multi-output GP.

``cov(f_p(x), f_q(x')) = B[p, q] * k(x, x')`` with ``B = W W^T + diag(kappa)`` (PSD by
construction). Outputs are observed at shared inputs with per-output noise. The marginal
likelihood is optimised over (W, log kappa, log noise) with central finite differences
(few parameters, so cheap) and the input kernel is held fixed.
"""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, _as2d
from .linalg import jitter_chol
from .optim import lbfgs


class ICMGP:
    def __init__(self, kernel: Kernel, n_outputs: int, rank: int = 1, seed: int = 0):
        self.k, self.P, self.R = kernel, int(n_outputs), int(rank)
        rng = np.random.default_rng(seed)
        self.W = 0.5 * rng.standard_normal((self.P, self.R))
        self.log_kappa = np.full(self.P, -2.0)
        self.log_noise = np.full(self.P, -2.0)

    @property
    def B(self) -> np.ndarray:
        return self.W @ self.W.T + np.diag(np.exp(self.log_kappa))

    def _pack(self):
        return np.concatenate([self.W.ravel(), self.log_kappa, self.log_noise])

    def _unpack(self, th):
        n = self.P * self.R
        self.W = th[:n].reshape(self.P, self.R)
        self.log_kappa = th[n:n + self.P]
        self.log_noise = th[n + self.P:]

    def _cov(self, Kx):
        n = Kx.shape[0]
        return np.kron(self.B, Kx) + np.kron(np.diag(np.exp(self.log_noise)), np.eye(n))

    def nlml(self, th=None) -> float:
        if th is not None:
            self._unpack(np.asarray(th, float))
        L = jitter_chol(self._cov(self.k(self.X)))
        z = np.linalg.solve(L, self._yv)
        return float(0.5 * z @ z + np.log(np.diag(L)).sum() + 0.5 * self._yv.size * np.log(2 * np.pi))

    def fit(self, X, Y, max_iter: int = 60):
        self.X = _as2d(X)
        Y = np.asarray(Y, float)
        assert Y.shape == (self.X.shape[0], self.P)
        self._ymean = Y.mean(0)
        self._yv = (Y - self._ymean).T.ravel()  # output-major stacking
        eps = 1e-5

        def fun(th):
            f = self.nlml(th)
            g = np.zeros_like(th)
            for i in range(th.size):
                d = np.zeros_like(th)
                d[i] = eps
                g[i] = (self.nlml(th + d) - self.nlml(th - d)) / (2 * eps)
            return f, g

        th, f, _ = lbfgs(fun, self._pack(), max_iter=max_iter)
        self._unpack(th)
        return self

    def predict(self, Xs, return_var: bool = False):
        """Mean (m, P) of the latent outputs; variance (m, P) if requested."""
        Xs = _as2d(Xs)
        n, m = self.X.shape[0], Xs.shape[0]
        Kx, Ks = self.k(self.X), self.k(Xs, self.X)
        L = jitter_chol(self._cov(Kx))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self._yv))
        Kstar = np.kron(self.B, Ks)  # (P m, P n)
        mean = (Kstar @ alpha).reshape(self.P, m).T + self._ymean
        if not return_var:
            return mean
        V = np.linalg.solve(L, Kstar.T)
        prior = np.kron(np.diag(self.B), self.k.diag(Xs))
        var = (prior - (V ** 2).sum(0)).reshape(self.P, m).T
        return mean, np.maximum(var, 1e-12)
