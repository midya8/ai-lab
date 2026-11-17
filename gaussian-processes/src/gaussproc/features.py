"""Finite-feature approximations of a GP: random Fourier features (iid and orthogonal)
and the Nystrom map, with the Bayesian-linear-regression GP they induce."""
from __future__ import annotations

import numpy as np

from .kernels import Kernel, Matern32, Matern52, RBF, _Stationary, _as2d
from .linalg import jitter_chol


class RandomFourierFeatures:
    """phi(x) = sqrt(2 sigma^2 / D) cos(W x + b) so that phi(x).phi(y) ~ k(x, y) (Rahimi & Recht).

    ``W`` is drawn from the kernel's spectral density: Gaussian for the RBF, multivariate
    Student-t with 2 nu degrees of freedom for Matern nu = 3/2 and 5/2.
    ``orthogonal=True`` uses blocks of orthogonal directions with chi-distributed norms
    (Yu et al. 2016), an unbiased estimator with lower variance."""

    def __init__(self, kernel: _Stationary, n_features: int, input_dim: int,
                 seed: int = 0, orthogonal: bool = False):
        if not isinstance(kernel, (RBF, Matern32, Matern52)):
            raise TypeError("RFF supports RBF, Matern32 and Matern52")
        rng = np.random.default_rng(seed)
        d, D = input_dim, n_features
        if orthogonal:
            blocks = []
            while sum(b.shape[0] for b in blocks) < D:
                G = rng.standard_normal((d, d))
                Q, R = np.linalg.qr(G)
                Q = Q * np.sign(np.diag(R))
                s = np.sqrt(rng.chisquare(d, size=d))
                blocks.append(s[:, None] * Q)
            Z = np.concatenate(blocks, 0)[:D]
        else:
            Z = rng.standard_normal((D, d))
        if isinstance(kernel, Matern32) or isinstance(kernel, Matern52):
            nu = 1.5 if isinstance(kernel, Matern32) else 2.5
            g = rng.chisquare(2 * nu, size=(D, 1))
            Z = Z / np.sqrt(g / (2 * nu))
        ls = kernel._ls(d)
        self.W = Z / ls
        self.b = rng.uniform(0, 2 * np.pi, D)
        self.scale = np.sqrt(2.0 * kernel.variance / D)
        self.n_features = D

    def __call__(self, X) -> np.ndarray:
        X = _as2d(X)
        return self.scale * np.cos(X @ self.W.T + self.b)


class NystromFeatures:
    """phi(x) = L_uu^-1 k(Z, x): phi(x).phi(y) = k(x, Z) K_uu^-1 k(Z, y)."""

    def __init__(self, kernel: Kernel, Z, jitter: float = 1e-8):
        self.kernel = kernel
        self.Z = _as2d(Z)
        self.L = jitter_chol(kernel(self.Z), jitter)
        self.n_features = self.Z.shape[0]

    def __call__(self, X) -> np.ndarray:
        return np.linalg.solve(self.L, self.kernel(self.Z, _as2d(X))).T


class FeatureGP:
    """Bayesian linear regression on features: ``f = phi(x) w``, ``w ~ N(0, I)``. Exact for
    the kernel ``phi phi^T``; cost O(n D^2)."""

    def __init__(self, features, noise: float = 0.1, center: bool = False):
        self.features = features
        self.noise = float(noise)
        self.center = center

    def fit(self, X, y) -> "FeatureGP":
        Phi = self.features(X)
        y = np.asarray(y, float).ravel()
        self.mean_ = float(y.mean()) if self.center else 0.0
        D = Phi.shape[1]
        P = Phi.T @ Phi / self.noise + np.eye(D)
        self._L = np.linalg.cholesky(P)
        self._Linv = np.linalg.inv(self._L)
        self._wmean = self._Linv.T @ (self._Linv @ (Phi.T @ (y - self.mean_) / self.noise))
        return self

    def predict(self, Xs, include_noise: bool = False):
        Ps = self.features(Xs)
        mean = Ps @ self._wmean + self.mean_
        V = Ps @ self._Linv.T
        var = np.maximum(np.sum(V * V, axis=1), 1e-12)
        if include_noise:
            var = var + self.noise
        return mean, var


def kernel_approx_error(K_true: np.ndarray, Phi: np.ndarray) -> float:
    """Relative Frobenius error of the feature kernel Phi Phi^T."""
    return float(np.linalg.norm(Phi @ Phi.T - K_true) / np.linalg.norm(K_true))
