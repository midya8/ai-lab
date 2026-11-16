"""Binary GP classification with a logistic likelihood and the Laplace approximation
(Rasmussen & Williams, Algs. 3.1-3.2): Newton mode finding in the numerically stable
``B = I + W^1/2 K W^1/2`` form, approximate log marginal likelihood, and predictive
probabilities by integrating the logistic against the Gaussian latent posterior with a
probit approximation. Hyper-parameters are fit by L-BFGS on central finite-difference
gradients of the approximate marginal likelihood."""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, _as2d
from .linalg import chol_solve, jitter_chol
from .optim import lbfgs


def _sigmoid(f):
    return 0.5 * (1.0 + np.tanh(0.5 * f))


def _log_lik(y01, f):
    # sum log sigmoid((2y-1) f), stable
    return float(-np.sum(np.logaddexp(0.0, -(2.0 * y01 - 1.0) * f)))


class LaplaceGPC:
    def __init__(self, kernel: Kernel, newton_tol: float = 1e-9, max_newton: int = 100):
        self.kernel = kernel
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.X = None

    def set_data(self, X, y):
        X = _as2d(X)
        y = np.asarray(y, float)
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("labels must be 0/1")
        self.X, self.y = X, y
        self._fit_mode()
        return self

    def _fit_mode(self):
        n = len(self.y)
        K = self.kernel(self.X)
        a = np.zeros(n)
        obj = lambda aa: -0.5 * aa @ (K @ aa) + _log_lik(self.y, K @ aa)
        cur = obj(a)
        for _ in range(self.max_newton):
            f = K @ a
            p = _sigmoid(f)
            W = p * (1 - p)
            sW = np.sqrt(W)
            L = jitter_chol(np.eye(n) + sW[:, None] * K * sW[None, :], 1e-12)
            b = W * f + (self.y - p)
            a_new = b - sW * chol_solve(L, sW * (K @ b))
            step = 1.0
            while step > 1e-8:  # damp until the penalised log-likelihood improves
                cand = a + step * (a_new - a)
                val = obj(cand)
                if val >= cur - 1e-12:
                    break
                step *= 0.5
            done = abs(val - cur) < self.newton_tol
            a, cur = cand, val
            if done:
                break
        f = K @ a
        p = _sigmoid(f)
        W = p * (1 - p)
        sW = np.sqrt(W)
        self.L = jitter_chol(np.eye(n) + sW[:, None] * K * sW[None, :], 1e-12)
        self.f_hat, self.W, self.sW, self.K, self.a = f, W, sW, K, a
        self.grad_ll = self.y - p

    # --- marginal likelihood --------------------------------------------------------
    def log_marginal_likelihood(self) -> float:
        psi = -0.5 * self.a @ self.f_hat + _log_lik(self.y, self.f_hat)
        return float(psi - np.sum(np.log(np.diag(self.L))))

    def fit(self, X, y, restarts: int = 3, seed: int = 0):
        X = _as2d(X)
        y = np.asarray(y, float)
        rng = np.random.default_rng(seed)
        theta0 = self.kernel.params.copy()

        def nlml(th):
            self.kernel.params = th
            return -self.set_data(X, y).log_marginal_likelihood()

        def fun(th):
            v = nlml(th)
            g = np.zeros_like(th)
            for i in range(th.size):
                e = np.zeros_like(th); e[i] = 1e-4
                g[i] = (nlml(th + e) - nlml(th - e)) / 2e-4
            return v, g

        best = (np.inf, theta0)
        for r in range(restarts):
            th0 = theta0 if r == 0 else theta0 + rng.normal(0, 0.7, theta0.size)
            th, f, _ = lbfgs(fun, th0, max_iter=40, bounds=(-4.0, 4.0))
            if f < best[0]:
                best = (f, th)
        self.kernel.params = best[1]
        self.set_data(X, y)
        return self

    # --- prediction -----------------------------------------------------------------
    def latent(self, Xs):
        """Mean and variance of the latent function at ``Xs``."""
        Xs = _as2d(Xs)
        Ks = self.kernel(self.X, Xs)
        mean = Ks.T @ self.grad_ll
        v = np.linalg.solve(self.L, self.sW[:, None] * Ks)
        var = self.kernel.diag(Xs) - (v * v).sum(0)
        return mean, np.maximum(var, 1e-12)

    def predict_proba(self, Xs, method: str = "probit"):
        m, v = self.latent(Xs)
        if method == "plugin":
            return _sigmoid(m)
        return _sigmoid(m / np.sqrt(1.0 + np.pi * v / 8.0))

    def predict_proba_quad(self, Xs, n: int = 80):
        """Reference: Gauss-Hermite integration of the logistic over the latent Gaussian."""
        m, v = self.latent(Xs)
        z, w = np.polynomial.hermite_e.hermegauss(n)
        return (w[None, :] / np.sqrt(2 * np.pi) * _sigmoid(m[:, None] + np.sqrt(v)[:, None] * z[None, :])).sum(1)


def log_loss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(p, y, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
