"""Student-t GP regression via EM on the Gaussian scale-mixture representation.

``y_i = f(x_i) + eps_i`` with ``eps_i ~ N(0, s2 / w_i)`` and ``w_i ~ Gamma(nu/2, nu/2)``
marginalises to a Student-t likelihood. EM alternates an exact Gaussian GP posterior with
per-point noise ``s2 / w_i`` (E-step for f) and the closed-form weight update
``E[w_i] = (nu + 1) / (nu + ((y_i - m_i)^2 + v_i) / s2)`` where ``m_i, v_i`` are the
posterior mean/variance of ``f(x_i)``. Kernel hyper-parameters are held fixed.
"""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, _as2d
from .linalg import jitter_chol


class RobustGP:
    def __init__(self, kernel: Kernel, noise: float = 0.05, nu: float = 3.0):
        if nu <= 0 or noise <= 0:
            raise ValueError("nu and noise must be positive")
        self.kernel, self.noise, self.nu = kernel, float(noise), float(nu)

    def _posterior_at_train(self, w):
        n = self.X.shape[0]
        K = self.kernel(self.X)
        L = jitter_chol(K + np.diag(self.noise / w))
        Linv = np.linalg.inv(L)
        Kinv = Linv.T @ Linv
        alpha = Kinv @ self.y
        m = K @ alpha
        # posterior cov of f at train = K - K Kinv K
        A = Linv @ K
        v = np.maximum(np.diag(K) - np.sum(A * A, axis=1), 1e-12)
        return m, v, alpha, Linv

    def fit(self, X, y, n_iter: int = 50, tol: float = 1e-6):
        self.X = _as2d(X)
        self.y = np.asarray(y, float).ravel()
        w = np.ones(self.y.size)
        for it in range(n_iter):
            m, v, alpha, Linv = self._posterior_at_train(w)
            w_new = (self.nu + 1.0) / (self.nu + ((self.y - m) ** 2 + v) / self.noise)
            delta = float(np.max(np.abs(w_new - w)))
            w = w_new
            if delta < tol:
                break
        self.n_iter_ = it + 1
        self.weights_ = w
        _, _, self._alpha, self._Linv = self._posterior_at_train(w)
        return self

    def predict(self, Xs, include_noise: bool = False):
        Xs = _as2d(Xs)
        Ks = self.kernel(Xs, self.X)
        mean = Ks @ self._alpha
        V = Ks @ self._Linv.T
        var = np.maximum(self.kernel.diag(Xs) - np.sum(V * V, axis=1), 1e-12)
        return mean, var + (self.noise if include_noise else 0.0)

    def outliers(self, threshold: float = 0.3) -> np.ndarray:
        """Indices of training points whose EM weight fell below ``threshold``."""
        return np.flatnonzero(self.weights_ < threshold)
