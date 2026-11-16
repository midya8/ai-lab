"""Covariance functions with analytic gradients with respect to log-hyper-parameters.

Every kernel exposes ``params`` (log scale), ``__call__(X, Y)``, ``diag(X)`` and
``grad(X)``: the list of ``dK(X, X)/d(log param_i)``. Sums and products compose and
their gradients follow the sum/product rules, so a structured kernel such as
``Periodic * RBF + Linear`` is fit with exact gradients too.
"""
from __future__ import annotations

import numpy as np

from .linalg import sqdist


def _as2d(X) -> np.ndarray:
    X = np.asarray(X, float)
    return X[:, None] if X.ndim == 1 else X


class Kernel:
    n_params = 0

    @property
    def params(self) -> np.ndarray:
        raise NotImplementedError

    @params.setter
    def params(self, theta):
        raise NotImplementedError

    def __call__(self, X, Y=None) -> np.ndarray:
        raise NotImplementedError

    def diag(self, X) -> np.ndarray:
        raise NotImplementedError

    def grad(self, X) -> list[np.ndarray]:
        raise NotImplementedError

    def param_names(self) -> list[str]:
        return [f"p{i}" for i in range(self.n_params)]

    def copy(self) -> "Kernel":
        import copy
        return copy.deepcopy(self)

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)


class _Stationary(Kernel):
    """sigma^2 * k(r), r^2 = sum_d (x_d - y_d)^2 / l_d^2, iso or ARD lengthscales."""

    def __init__(self, lengthscale=1.0, variance=1.0, ard_dims: int | None = None):
        ls = np.atleast_1d(np.asarray(lengthscale, float))
        if ard_dims is not None and ls.size == 1:
            ls = np.full(ard_dims, float(ls[0]))
        self._log_ls = np.log(ls)
        self._log_var = float(np.log(variance))

    @property
    def n_params(self):  # type: ignore[override]
        return self._log_ls.size + 1

    @property
    def lengthscale(self) -> np.ndarray:
        return np.exp(self._log_ls)

    @property
    def variance(self) -> float:
        return float(np.exp(self._log_var))

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self._log_ls, [self._log_var]])

    @params.setter
    def params(self, theta):
        theta = np.asarray(theta, float)
        self._log_ls = theta[:-1].copy()
        self._log_var = float(theta[-1])

    def param_names(self):
        k = self._log_ls.size
        names = ["log_lengthscale"] if k == 1 else [f"log_lengthscale_{i}" for i in range(k)]
        return names + ["log_variance"]

    def _ls(self, d: int) -> np.ndarray:
        return np.full(d, self.lengthscale[0]) if self._log_ls.size == 1 else self.lengthscale

    def _r2(self, X, Y):
        return sqdist(X, Y, self._ls(X.shape[1]))

    def _unit(self, r2) -> np.ndarray:           # k(r) with sigma^2 = 1
        raise NotImplementedError

    def _dunit(self, r2) -> np.ndarray:          # factor G with dK/dlog l_d = sigma^2 * G * (dx_d^2 / l_d^2)
        raise NotImplementedError

    def __call__(self, X, Y=None):
        X = _as2d(X)
        Y = X if Y is None else _as2d(Y)
        return self.variance * self._unit(self._r2(X, Y))

    def diag(self, X):
        return np.full(_as2d(X).shape[0], self.variance)

    def grad(self, X):
        X = _as2d(X)
        d = X.shape[1]
        ls = self._ls(d)
        r2 = sqdist(X, X, ls)
        G = self.variance * self._dunit(r2)
        out = []
        if self._log_ls.size == 1:
            out.append(G * r2)                    # sum_d dx_d^2/l^2 = r^2
        else:
            for k in range(d):
                diff2 = (X[:, k][:, None] - X[:, k][None, :]) ** 2 / ls[k] ** 2
                out.append(G * diff2)
        out.append(self.variance * self._unit(r2))
        return out


class RBF(_Stationary):
    def _unit(self, r2):
        return np.exp(-0.5 * r2)

    def _dunit(self, r2):
        return np.exp(-0.5 * r2)


class Matern32(_Stationary):
    def _unit(self, r2):
        r = np.sqrt(r2)
        return (1.0 + np.sqrt(3.0) * r) * np.exp(-np.sqrt(3.0) * r)

    def _dunit(self, r2):
        return 3.0 * np.exp(-np.sqrt(3.0) * np.sqrt(r2))


class Matern52(_Stationary):
    def _unit(self, r2):
        r = np.sqrt(r2)
        return (1.0 + np.sqrt(5.0) * r + 5.0 * r2 / 3.0) * np.exp(-np.sqrt(5.0) * r)

    def _dunit(self, r2):
        r = np.sqrt(r2)
        return (5.0 / 3.0) * (1.0 + np.sqrt(5.0) * r) * np.exp(-np.sqrt(5.0) * r)


class Periodic(Kernel):
    """sigma^2 exp(-2 sum_d sin^2(pi (x_d - y_d) / p) / l^2): a product of 1-D periodic kernels,
    one per input dimension, sharing a period. (The version on the Euclidean distance is not
    positive semi-definite for dimension > 1.)"""

    n_params = 3

    def __init__(self, lengthscale=1.0, period=1.0, variance=1.0):
        self._t = np.log([lengthscale, period, variance]).astype(float)

    @property
    def params(self):
        return self._t.copy()

    @params.setter
    def params(self, theta):
        self._t = np.asarray(theta, float).copy()

    @property
    def lengthscale(self):
        return float(np.exp(self._t[0]))

    @property
    def period(self):
        return float(np.exp(self._t[1]))

    @property
    def variance(self):
        return float(np.exp(self._t[2]))

    def param_names(self):
        return ["log_lengthscale", "log_period", "log_variance"]

    def _u(self, X, Y):
        return np.pi * (X[:, None, :] - Y[None, :, :]) / self.period     # n x m x d

    def __call__(self, X, Y=None):
        X = _as2d(X)
        Y = X if Y is None else _as2d(Y)
        S = (np.sin(self._u(X, Y)) ** 2).sum(-1)
        return self.variance * np.exp(-2.0 * S / self.lengthscale ** 2)

    def diag(self, X):
        return np.full(_as2d(X).shape[0], self.variance)

    def grad(self, X):
        X = _as2d(X)
        u = self._u(X, X)
        s, c = np.sin(u), np.cos(u)
        S = (s * s).sum(-1)
        k = self.variance * np.exp(-2.0 * S / self.lengthscale ** 2)
        d_ls = k * 4.0 * S / self.lengthscale ** 2
        d_p = k * 4.0 * (u * s * c).sum(-1) / self.lengthscale ** 2
        return [d_ls, d_p, k]


class Linear(Kernel):
    """sigma^2 x.y  (Bayesian linear regression prior; non-stationary)."""

    n_params = 1

    def __init__(self, variance=1.0):
        self._t = np.array([np.log(variance)])

    @property
    def params(self):
        return self._t.copy()

    @params.setter
    def params(self, theta):
        self._t = np.asarray(theta, float).copy()

    @property
    def variance(self):
        return float(np.exp(self._t[0]))

    def param_names(self):
        return ["log_variance"]

    def __call__(self, X, Y=None):
        X = _as2d(X)
        Y = X if Y is None else _as2d(Y)
        return self.variance * X @ Y.T

    def diag(self, X):
        X = _as2d(X)
        return self.variance * (X * X).sum(1)

    def grad(self, X):
        X = _as2d(X)
        return [self.variance * X @ X.T]


class _Composite(Kernel):
    def __init__(self, a: Kernel, b: Kernel):
        self.a, self.b = a, b

    @property
    def n_params(self):  # type: ignore[override]
        return self.a.n_params + self.b.n_params

    @property
    def params(self):
        return np.concatenate([self.a.params, self.b.params])

    @params.setter
    def params(self, theta):
        theta = np.asarray(theta, float)
        self.a.params = theta[: self.a.n_params]
        self.b.params = theta[self.a.n_params:]

    def param_names(self):
        return [f"a.{n}" for n in self.a.param_names()] + [f"b.{n}" for n in self.b.param_names()]


class Sum(_Composite):
    def __call__(self, X, Y=None):
        return self.a(X, Y) + self.b(X, Y)

    def diag(self, X):
        return self.a.diag(X) + self.b.diag(X)

    def grad(self, X):
        return self.a.grad(X) + self.b.grad(X)


class Product(_Composite):
    def __call__(self, X, Y=None):
        return self.a(X, Y) * self.b(X, Y)

    def diag(self, X):
        return self.a.diag(X) * self.b.diag(X)

    def grad(self, X):
        Ka, Kb = self.a(X), self.b(X)
        return [g * Kb for g in self.a.grad(X)] + [Ka * g for g in self.b.grad(X)]


class SpectralMixture(Kernel):
    """Spectral-mixture kernel (Wilson & Adams 2013), product over input dimensions:
    k(t) = sum_q w_q prod_d exp(-2 pi^2 t_d^2 v_qd) cos(2 pi t_d mu_qd).
    The spectral density is a Gaussian mixture, so it can express several periodicities and
    extrapolate them. Parameters (all log): w (Q), mu (Q*d), v (Q*d). Multimodal likelihood:
    initialise means from the data periodogram or use restarts."""

    def __init__(self, weights, means, variances):
        w = np.atleast_1d(np.asarray(weights, float))
        mu = np.asarray(means, float).reshape(w.size, -1)
        v = np.asarray(variances, float).reshape(w.size, -1)
        assert mu.shape == v.shape
        self.Q, self.d = mu.shape
        self._t = np.log(np.concatenate([w, mu.ravel(), v.ravel()]))

    @property
    def n_params(self):  # type: ignore[override]
        return self.Q * (1 + 2 * self.d)

    @property
    def params(self):
        return self._t.copy()

    @params.setter
    def params(self, theta):
        self._t = np.asarray(theta, float).copy()

    @property
    def weights(self):
        return np.exp(self._t[: self.Q])

    @property
    def means(self):
        return np.exp(self._t[self.Q: self.Q + self.Q * self.d]).reshape(self.Q, self.d)

    @property
    def variances(self):
        return np.exp(self._t[self.Q + self.Q * self.d:]).reshape(self.Q, self.d)

    def param_names(self):
        return ([f"log_w{q}" for q in range(self.Q)]
                + [f"log_mu{q}_{j}" for q in range(self.Q) for j in range(self.d)]
                + [f"log_v{q}_{j}" for q in range(self.Q) for j in range(self.d)])

    def _parts(self, X, Y):
        T = X[:, None, :] - Y[None, :, :]                      # n x m x d
        w, mu, v = self.weights, self.means, self.variances
        env = np.exp(-2 * np.pi ** 2 * T[:, :, None, :] ** 2 * v[None, None])    # n m Q d
        osc = np.cos(2 * np.pi * T[:, :, None, :] * mu[None, None])
        return T, w, mu, v, env, osc

    def __call__(self, X, Y=None):
        X = _as2d(X)
        Y = X if Y is None else _as2d(Y)
        _, w, _, _, env, osc = self._parts(X, Y)
        return (w[None, None] * (env * osc).prod(-1)).sum(-1)

    def diag(self, X):
        return np.full(_as2d(X).shape[0], self.weights.sum())

    def grad(self, X):
        X = _as2d(X)
        T, w, mu, v, env, osc = self._parts(X, X)
        f = env * osc                                          # n m Q d
        prod = f.prod(-1)                                      # n m Q
        out = [w[q] * prod[:, :, q] for q in range(self.Q)]
        Tq = T[:, :, None, :]
        d_mu = -env * np.sin(2 * np.pi * Tq * mu[None, None]) * 2 * np.pi * Tq * mu[None, None]
        d_v = f * (-2 * np.pi ** 2 * Tq ** 2 * v[None, None])
        for dd in (d_mu, d_v):
            for q in range(self.Q):
                for j in range(self.d):
                    g = dd[:, :, q, j] * np.prod(np.delete(f[:, :, q, :], j, axis=-1), axis=-1)
                    out.append(w[q] * g)
        return out
