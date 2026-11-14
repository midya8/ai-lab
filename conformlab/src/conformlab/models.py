"""Small base learners (NumPy only): random-Fourier-feature ridge, quantile regression, softmax."""
from __future__ import annotations

import numpy as np


class RFF:
    """Random Fourier features of an RBF kernel plus a constant column (the intercept)."""

    def __init__(self, d, D=120, lengthscale=1.0, seed=0):
        r = np.random.default_rng(seed)
        self.W = r.normal(0, 1.0 / lengthscale, size=(d, D))
        self.b = r.uniform(0, 2 * np.pi, size=D)

    def __call__(self, X):
        X = np.asarray(X, float)
        Z = np.cos(X @ self.W + self.b)
        return np.hstack([Z, np.ones((len(X), 1))])


class RidgeRegressor:
    """Closed-form ridge on a feature map; the intercept column is (almost) unpenalised."""

    def __init__(self, feat, lam=1.0):
        self.feat, self.lam, self.w = feat, lam, None

    def fit(self, X, y):
        P = self.feat(X)
        pen = np.full(P.shape[1], self.lam)
        pen[-1] = 1e-8
        self.w = np.linalg.solve(P.T @ P + np.diag(pen), P.T @ np.asarray(y, float))
        return self

    def predict(self, X):
        return self.feat(X) @ self.w


class QuantileRegressor:
    """Linear-in-features conditional quantile via pinball loss + L2, full-batch Adam."""

    def __init__(self, feat, tau, lam=1e-3, steps=1500, lr=0.03):
        self.feat, self.tau, self.lam, self.steps, self.lr = feat, tau, lam, steps, lr
        self.w = None

    def fit(self, X, y):
        P = self.feat(X)
        y = np.asarray(y, float)
        n, p = P.shape
        w = np.zeros(p)
        w[-1] = np.quantile(y, self.tau)
        m, v = np.zeros(p), np.zeros(p)
        pen = np.full(p, self.lam)
        pen[-1] = 0.0
        for t in range(1, self.steps + 1):
            r = y - P @ w
            g = P.T @ np.where(r > 0, -self.tau, 1 - self.tau) / n + pen * w
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            lr = self.lr * (0.5 * (1 + np.cos(np.pi * t / self.steps)) * 0.95 + 0.05)
            w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.w = w
        return self

    def predict(self, X):
        return self.feat(X) @ self.w


class SoftmaxRegressor:
    """Multinomial logistic regression on standardised inputs (+ optional feature map), Adam."""

    def __init__(self, K, lam=1e-3, steps=400, lr=0.05, feat=None):
        self.K, self.lam, self.steps, self.lr, self.feat = K, lam, steps, lr, feat
        self.W = None

    def _phi(self, X):
        X = (np.asarray(X, float) - self.mu) / self.sd
        if self.feat is not None:
            X = self.feat(X)[:, :-1]
        return np.hstack([X, np.ones((len(X), 1))])

    def fit(self, X, y):
        X = np.asarray(X, float)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        P = self._phi(X)
        Y = np.eye(self.K)[np.asarray(y, int)]
        n, p = P.shape
        W = np.zeros((p, self.K))
        m, v = np.zeros_like(W), np.zeros_like(W)
        pen = np.full((p, 1), self.lam)
        pen[-1] = 0.0
        for t in range(1, self.steps + 1):
            G = P.T @ (_softmax(P @ W) - Y) / n + pen * W
            m = 0.9 * m + 0.1 * G
            v = 0.999 * v + 0.001 * G * G
            W = W - self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        self.W = W
        return self

    def predict_proba(self, X):
        return _softmax(self._phi(X) @ self.W)


def _softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)
