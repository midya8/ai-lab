"""Stochastic variational GP regression (Hensman et al., 2013) with natural-gradient steps.

q(u) is kept in the whitened parametrisation u = L_uu v, q(v) = N(m, S), prior N(0, I).
For a Gaussian likelihood the ELBO is conjugate in (m, S), so a natural-gradient step of size
rho on a minibatch is just a convex blend of the current natural parameters with the
minibatch-estimated optimum:

    P*  = I + (n/b) / s2 * A_b A_b^T          (precision,   A = L_uu^-1 K_uf)
    h*  =     (n/b) / s2 * A_b y_b            (P m)
    P <- (1-rho) P + rho P*,   h <- (1-rho) h + rho h*

With rho = 1 and the full batch this is the Titsias (2009) optimum, so SVGP converges to the
collapsed VFE predictor while only ever touching ``batch`` points per step: O(b m^2 + m^3).
"""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, _as2d
from .linalg import jitter_chol


class SVGP:
    def __init__(self, kernel: Kernel, Z, noise: float = 0.1, jitter: float = 1e-8, seed: int = 0):
        self.kernel = kernel
        self.Z = _as2d(Z).copy()
        self.noise = float(noise)
        self.jitter = jitter
        self.rng = np.random.default_rng(seed)
        m = self.Z.shape[0]
        self._Luu = jitter_chol(self.kernel(self.Z), jitter)
        self.P = np.eye(m)          # S^-1
        self.h = np.zeros(m)        # S^-1 m
        self.steps = 0

    def _A(self, X):
        return np.linalg.solve(self._Luu, self.kernel(self.Z, X))

    def step(self, Xb, yb, n_total: int, rho: float):
        """One natural-gradient step on a minibatch (0 < rho <= 1)."""
        if not 0.0 < rho <= 1.0:
            raise ValueError("rho must be in (0, 1]")
        Xb = _as2d(Xb)
        yb = np.asarray(yb, float).ravel() - self.mean_
        A = self._A(Xb)
        s = (n_total / Xb.shape[0]) / self.noise
        self.P = (1 - rho) * self.P + rho * (np.eye(self.P.shape[0]) + s * A @ A.T)
        self.h = (1 - rho) * self.h + rho * (s * A @ yb)
        self.steps += 1

    def fit(self, X, y, epochs: int = 30, batch: int = 32, lr: float = 0.5, decay: float = 0.6):
        """Epoch-based training with a Robbins-Monro schedule rho_t = lr * (1 + t)^-decay
        (decay in (0.5, 1] satisfies the usual conditions)."""
        X = _as2d(X)
        y = np.asarray(y, float).ravel()
        self.mean_ = float(y.mean())
        n = X.shape[0]
        t = 0
        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                self.step(X[idx], y[idx], n, min(1.0, lr * (1 + t) ** -decay))
                t += 1
        return self

    def elbo(self, X, y) -> float:
        """Full-data ELBO at the current q (whitened)."""
        X = _as2d(X)
        y = np.asarray(y, float).ravel() - self.mean_
        n, m = X.shape[0], self.P.shape[0]
        A = self._A(X)
        S = np.linalg.inv(self.P)
        mu = S @ self.h
        f = A.T @ mu
        s2 = self.noise
        qdiag = np.sum(A * A, axis=0)
        trS = np.einsum("ij,jk,ik->i", A.T, S, A.T)
        exp_ll = -0.5 * (n * np.log(2 * np.pi * s2) + (np.sum((y - f) ** 2) +
                         np.sum(self.kernel.diag(X) - qdiag) + trS.sum()) / s2)
        _, logdetS = np.linalg.slogdet(S)
        kl = 0.5 * (np.trace(S) + mu @ mu - m - logdetS)
        return float(exp_ll - kl)

    def predict(self, Xs, include_noise: bool = False):
        Xs = _as2d(Xs)
        W = self._A(Xs)
        S = np.linalg.inv(self.P)
        mu = S @ self.h
        mean = W.T @ mu + self.mean_
        var = self.kernel.diag(Xs) - np.sum(W * W, axis=0) + np.einsum("ij,jk,ki->i", W.T, S, W)
        var = np.maximum(var, 1e-12)
        return mean, (var + self.noise if include_noise else var)
