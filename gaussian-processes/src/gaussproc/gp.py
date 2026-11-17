"""Exact Gaussian-process regression: marginal likelihood with analytic gradients,
closed-form leave-one-out, posterior sampling and multi-restart hyper-parameter fitting."""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, _as2d
from .linalg import jitter_chol
from .optim import lbfgs

LOG2PI = float(np.log(2.0 * np.pi))


class ExactGP:
    """GP regression ``y = f(x) + eps`` with ``f ~ GP(mean, k)`` and ``eps ~ N(0, noise)``.

    ``params`` is ``[kernel log-params..., log noise variance]``. The mean is a constant
    that can be fit as the sample mean (``center=True``) or fixed at 0.
    """

    def __init__(self, kernel: Kernel, noise: float = 0.1, center: bool = False):
        self.kernel = kernel
        self.log_noise = float(np.log(noise))
        self.center = center
        self.X = None
        self.y = None
        self.mean_ = 0.0

    # --- parameters ---------------------------------------------------------------
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

    # --- data ----------------------------------------------------------------------
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
        n = self.X.shape[0]
        K = self.kernel(self.X) + self.noise * np.eye(n)
        self._L = jitter_chol(K)
        self._Linv = np.linalg.inv(self._L)
        self._Kinv = self._Linv.T @ self._Linv
        self._alpha = self._Kinv @ self.y

    # --- marginal likelihood -------------------------------------------------------
    def log_marginal_likelihood(self) -> float:
        n = self.y.size
        return float(-0.5 * self.y @ self._alpha - np.sum(np.log(np.diag(self._L))) - 0.5 * n * LOG2PI)

    def lml_and_grad(self) -> tuple[float, np.ndarray]:
        """Value and analytic gradient w.r.t. ``params`` (all log-scale).

        d lml / d theta = 1/2 tr((alpha alpha^T - K^-1) dK/dtheta)."""
        W = np.outer(self._alpha, self._alpha) - self._Kinv
        g = [0.5 * float(np.sum(W * dK)) for dK in self.kernel.grad(self.X)]
        g.append(0.5 * self.noise * float(np.trace(W)))
        return self.log_marginal_likelihood(), np.array(g)

    # --- fitting -------------------------------------------------------------------
    def _neg(self, theta):
        self.params = theta
        v, g = self.lml_and_grad()
        return -v, -g

    def fit(self, X, y, restarts: int = 0, seed: int = 0, max_iter: int = 100,
            init_scale: float = 1.5) -> "ExactGP":
        """Maximise the marginal likelihood with L-BFGS from the current parameters plus
        ``restarts`` random log-space perturbations; keeps the best optimum.
        ``self.fit_log_`` records the lml reached by every start."""
        self.set_data(X, y)
        start = self.params.copy()
        rng = np.random.default_rng(seed)
        starts = [start] + [start + rng.normal(0, init_scale, start.shape) for _ in range(restarts)]
        results = []
        for th0 in starts:
            try:
                th, f, _ = lbfgs(self._neg, th0, max_iter=max_iter)
            except np.linalg.LinAlgError:
                continue
            results.append((-f, th))
        self.fit_log_ = [float(r[0]) for r in results]
        best = max(results, key=lambda r: r[0])
        self.params = best[1]
        return self

    # --- prediction ----------------------------------------------------------------
    def predict(self, Xs, include_noise: bool = False):
        """Posterior predictive mean and variance of f (or of y with ``include_noise``)."""
        Xs = _as2d(Xs)
        Ks = self.kernel(Xs, self.X)
        mean = Ks @ self._alpha + self.mean_
        V = Ks @ self._Linv.T                      # rows: L^-1 k_*
        var = self.kernel.diag(Xs) - np.sum(V * V, axis=1)
        var = np.maximum(var, 1e-12)
        if include_noise:
            var = var + self.noise
        return mean, var

    def predict_cov(self, Xs) -> tuple[np.ndarray, np.ndarray]:
        Xs = _as2d(Xs)
        Ks = self.kernel(Xs, self.X)
        V = Ks @ self._Linv.T
        return Ks @ self._alpha + self.mean_, self.kernel(Xs) - V @ V.T

    def sample_posterior(self, Xs, n_samples: int = 5, seed: int = 0) -> np.ndarray:
        mean, cov = self.predict_cov(Xs)
        L = jitter_chol(cov, jitter=1e-8)
        z = np.random.default_rng(seed).standard_normal((len(mean), n_samples))
        return (mean[:, None] + L @ z).T

    def sample_prior(self, X, n_samples: int = 1, seed: int = 0, include_noise: bool = True):
        X = _as2d(X)
        K = self.kernel(X)
        L = jitter_chol(K, jitter=1e-8)
        z = np.random.default_rng(seed).standard_normal((X.shape[0], n_samples))
        f = (L @ z).T
        if include_noise:
            f = f + np.sqrt(self.noise) * np.random.default_rng(seed + 1).standard_normal(f.shape)
        return f

    # --- closed-form leave-one-out ---------------------------------------------------
    def loo(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out predictive mean and variance of y_i for all i without refitting
        (Rasmussen & Williams 5.4.2): mu_i = y_i - [K^-1 y]_i / [K^-1]_ii, var_i = 1 / [K^-1]_ii."""
        d = np.diag(self._Kinv)
        return self.y + self.mean_ - self._alpha / d, 1.0 / d

    def loo_log_predictive(self) -> float:
        mu, var = self.loo()
        r = self.y + self.mean_ - mu
        return float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * r * r / var))


def gaussian_nlpd(y, mean, var) -> float:
    """Mean negative log predictive density under independent Gaussians."""
    y, mean, var = np.asarray(y), np.asarray(mean), np.asarray(var)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def coverage(y, mean, var, level: float = 0.95) -> float:
    z = {0.5: 0.6744897502, 0.8: 1.2815515655, 0.9: 1.6448536270, 0.95: 1.9599639845, 0.99: 2.5758293035}[level]
    return float(np.mean(np.abs(np.asarray(y) - mean) <= z * np.sqrt(var)))
