"""Most-likely heteroscedastic GP (Kersting et al., 2007), fixed kernels.

Model: ``y = f(x) + eps(x)``, ``eps ~ N(0, exp(g(x)))`` with GPs on ``f`` and on the
log-noise ``g``. Alternate (a) a GP on ``f`` with per-point noise ``exp(g)`` and (b) a GP on
``log r^2`` where ``r`` are residuals against the *leave-one-out* mean of (a) (LOO avoids the
shrinkage that makes in-sample residuals too small). ``E[log chi2_1] = -1.2704`` is added back
so ``exp(g)`` is on the variance scale.
"""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, _as2d
from .linalg import jitter_chol

LOG_CHI2_MEAN = -1.2703628454614782  # digamma(1/2) + log 2


class HeteroGP:
    def __init__(self, kernel_f: Kernel, kernel_g: Kernel, g_noise: float = 0.2):
        self.kf, self.kg, self.g_noise = kernel_f, kernel_g, float(g_noise)

    def _solve(self, kernel, X, y, noise_vec):
        K = kernel(X)
        L = jitter_chol(K + np.diag(noise_vec))
        Linv = np.linalg.inv(L)
        Kinv = Linv.T @ Linv
        return Kinv, Kinv @ y

    def fit(self, X, y, n_iter: int = 8, init_noise: float = 0.1):
        self.X = _as2d(X)
        self.y = np.asarray(y, float).ravel()
        n = self.y.size
        noise = np.full(n, float(init_noise))
        for _ in range(n_iter):
            Kinv, alpha = self._solve(self.kf, self.X, self.y, noise)
            # closed-form LOO residual: (Kinv y)_i / (Kinv)_ii, LOO variance 1/(Kinv)_ii - noise_i
            res = alpha / np.diag(Kinv)
            z = np.log(np.maximum(res ** 2, 1e-10)) - LOG_CHI2_MEAN
            zc = z - z.mean()
            Kg_inv, a_g = self._solve(self.kg, self.X, zc, np.full(n, self.g_noise))
            g = z.mean() + self.kg(self.X) @ a_g
            new = np.exp(g)
            if np.max(np.abs(np.log(new / noise))) < 1e-4:
                noise = new
                break
            noise = new
        self.noise_ = noise
        self._g_mean = float(z.mean())
        self._Kg_inv, self._a_g = Kg_inv, a_g
        self._Kinv, self._alpha = self._solve(self.kf, self.X, self.y, noise)
        return self

    def noise_at(self, Xs):
        Xs = _as2d(Xs)
        return np.exp(self._g_mean + self.kg(Xs, self.X) @ self._a_g)

    def predict(self, Xs, include_noise: bool = False):
        Xs = _as2d(Xs)
        Ks = self.kf(Xs, self.X)
        mean = Ks @ self._alpha
        var = np.maximum(self.kf.diag(Xs) - np.sum((Ks @ self._Kinv) * Ks, axis=1), 1e-12)
        if include_noise:
            var = var + self.noise_at(Xs)
        return mean, var
