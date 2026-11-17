"""Gaussian-process regression with ARD RBF / Matern-5/2 kernels and analytic marginal-likelihood gradients."""
from __future__ import annotations
import numpy as np

_SQRT5 = np.sqrt(5.0)
# bounds on log-hyperparameters (inputs are assumed to live in [0,1]^d, targets standardised)
LS_BOUNDS = (np.log(0.03), np.log(10.0))
SF_BOUNDS = (np.log(0.05), np.log(20.0))
SN_BOUNDS = (np.log(1e-6), np.log(1.0))


class Kernel:
    """k(x, x') = sf2 * g(r),  r^2 = sum_d (x_d - x'_d)^2 / l_d^2,  g = RBF or Matern-5/2."""

    def __init__(self, lengthscales, sf2=1.0, kind="matern52"):
        if kind not in ("rbf", "matern52"):
            raise ValueError("kind must be 'rbf' or 'matern52'")
        self.ls = np.atleast_1d(np.asarray(lengthscales, float))
        self.sf2 = float(sf2)
        self.kind = kind

    def _r2_parts(self, A, B):
        D = (A[:, None, :] - B[None, :, :]) ** 2 / self.ls**2     # (n, m, d) per-dimension scaled sq. distances
        return D

    def _g(self, r2):
        if self.kind == "rbf":
            return np.exp(-0.5 * r2)
        r = np.sqrt(np.maximum(r2, 0.0))
        return (1.0 + _SQRT5 * r + 5.0 / 3.0 * r2) * np.exp(-_SQRT5 * r)

    def __call__(self, A, B):
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        return self.sf2 * self._g(self._r2_parts(A, B).sum(-1))

    def dk_dx(self, Xs, X):
        """Gradient of k(x_i, X_j) w.r.t. x_i: array (n, m, d)."""
        Xs, X = np.atleast_2d(Xs), np.atleast_2d(X)
        diff = Xs[:, None, :] - X[None, :, :]
        r2 = (diff**2 / self.ls**2).sum(-1)
        if self.kind == "rbf":
            dg = -0.5 * np.exp(-0.5 * r2)
        else:
            r = np.sqrt(np.maximum(r2, 0.0))
            dg = -(5.0 / 6.0) * (1.0 + _SQRT5 * r) * np.exp(-_SQRT5 * r)
        return self.sf2 * dg[:, :, None] * 2.0 * diff / self.ls**2

    def grads(self, A):
        """K(A, A) and dK/d(log l_d) (list), dK/d(log sf2) for symmetric A."""
        D = self._r2_parts(A, A)
        r2 = D.sum(-1)
        K = self.sf2 * self._g(r2)
        if self.kind == "rbf":
            dls = [K * D[:, :, d] for d in range(D.shape[2])]
        else:
            r = np.sqrt(np.maximum(r2, 0.0))
            # d g / d log l_d = (5/3)(1 + sqrt5 r) exp(-sqrt5 r) * D_d   (no 1/r singularity)
            c = self.sf2 * (5.0 / 3.0) * (1.0 + _SQRT5 * r) * np.exp(-_SQRT5 * r)
            dls = [c * D[:, :, d] for d in range(D.shape[2])]
        return K, dls, K.copy()


def _chol(K):
    jitter = 0.0
    for _ in range(8):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(len(K)))
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 10
    raise np.linalg.LinAlgError("kernel matrix not positive definite")


def _solve_chol(L, B):
    return np.linalg.solve(L.T, np.linalg.solve(L, B))


def log_marginal_likelihood(X, y, kernel: Kernel, sn2: float, grad: bool = False):
    """log p(y | X) for a zero-mean GP; with grad=True also the gradient w.r.t. [log l_1..l_d, log sf2, log sn2]."""
    n = len(y)
    K0, dls, dsf = kernel.grads(X)
    K = K0 + sn2 * np.eye(n)
    L = _chol(K)
    alpha = _solve_chol(L, y)
    ll = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * n * np.log(2 * np.pi)
    if not grad:
        return ll
    Kinv = _solve_chol(L, np.eye(n))
    W = np.outer(alpha, alpha) - Kinv
    g = [0.5 * np.sum(W * dK) for dK in dls + [dsf]]
    g.append(0.5 * np.sum(np.diag(W)) * sn2)
    return ll, np.array(g)


class GP:
    """Zero-mean GP on standardised targets. Predictions are returned on the original y scale."""

    def __init__(self, kernel: Kernel, sn2: float = 1e-4, standardise: bool = True):
        self.kernel, self.sn2, self.standardise = kernel, float(sn2), standardise
        self.X = self.y = None

    def fit(self, X, y):
        X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float).ravel()
        self.X, self.y_raw = X, y
        self.mu_y = y.mean() if self.standardise else 0.0
        self.sd_y = (y.std() if y.std() > 0 else 1.0) if self.standardise else 1.0
        self.y = (y - self.mu_y) / self.sd_y
        K = self.kernel(X, X) + self.sn2 * np.eye(len(X))
        self.L = _chol(K)
        self.alpha = _solve_chol(self.L, self.y)
        return self

    def predict(self, Xs, return_cov=False):
        Xs = np.atleast_2d(Xs)
        Ks = self.kernel(Xs, self.X)
        mu = Ks @ self.alpha
        V = np.linalg.solve(self.L, Ks.T)
        if return_cov:
            cov = self.kernel(Xs, Xs) - V.T @ V
            return mu * self.sd_y + self.mu_y, cov * self.sd_y**2
        var = np.maximum(self.kernel.sf2 - (V**2).sum(0), 1e-12)
        return mu * self.sd_y + self.mu_y, np.sqrt(var) * self.sd_y

    def predict_grad(self, Xs):
        """Posterior mean, std and their analytic gradients w.r.t. each row of Xs (original y scale).
        Returns mu (n,), sd (n,), dmu (n, d), dsd (n, d)."""
        Xs = np.atleast_2d(Xs)
        Ks = self.kernel(Xs, self.X)                      # (n, m)
        dK = self.kernel.dk_dx(Xs, self.X)                # (n, m, d)
        V = np.linalg.solve(self.L, Ks.T)                 # (m, n)
        W = np.linalg.solve(self.L.T, V)                  # K^-1 k_*, (m, n)
        mu = Ks @ self.alpha
        var = np.maximum(self.kernel.sf2 - (V**2).sum(0), 1e-12)
        dmu = np.einsum("nmd,m->nd", dK, self.alpha)
        dvar = -2.0 * np.einsum("nmd,mn->nd", dK, W)
        sd = np.sqrt(var)
        dsd = dvar / (2.0 * sd[:, None])
        return mu * self.sd_y + self.mu_y, sd * self.sd_y, dmu * self.sd_y, dsd * self.sd_y

    def sample(self, Xs, n_samples, rng):
        """Joint posterior samples of the latent function at Xs: (n_samples, len(Xs))."""
        mu, cov = self.predict(Xs, return_cov=True)
        Lc = _chol(cov + 1e-9 * self.sd_y**2 * np.eye(len(mu)))
        return mu[None, :] + rng.standard_normal((n_samples, len(mu))) @ Lc.T

    def condition(self, Xnew, ynew):
        """Same hyperparameters, extra (X, y) pairs (used by constant-liar batching)."""
        g = GP(self.kernel, self.sn2, self.standardise)
        return g.fit(np.vstack([self.X, np.atleast_2d(Xnew)]), np.concatenate([self.y_raw, np.ravel(ynew)]))


def _pack(kernel, sn2):
    return np.concatenate([np.log(kernel.ls), [np.log(kernel.sf2), np.log(sn2)]])


def _clip(theta, d):
    lo = np.array([LS_BOUNDS[0]] * d + [SF_BOUNDS[0], SN_BOUNDS[0]])
    hi = np.array([LS_BOUNDS[1]] * d + [SF_BOUNDS[1], SN_BOUNDS[1]])
    return np.clip(theta, lo, hi)


def fit_gp(X, y, kind="matern52", n_starts=3, steps=120, lr=0.08, rng=None, init=None, noise_floor=1e-6):
    """Maximum-marginal-likelihood fit by multi-start Adam in log-space (analytic gradients, box-clipped).
    `init` is an optional (lengthscales, sf2, sn2) warm start. Returns a fitted GP."""
    rng = rng or np.random.default_rng(0)
    X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float).ravel()
    d = X.shape[1]
    sd = y.std() if y.std() > 0 else 1.0
    ys = (y - y.mean()) / sd
    starts = [np.concatenate([np.full(d, np.log(0.3)), [0.0, np.log(1e-2)]])]
    if init is not None:
        starts.insert(0, np.concatenate([np.log(init[0]), [np.log(init[1]), np.log(init[2])]]))
    while len(starts) < n_starts:
        starts.append(np.concatenate([rng.uniform(np.log(0.1), np.log(2.0), d), [rng.uniform(-1, 1), rng.uniform(np.log(1e-4), np.log(0.1))]]))
    best, best_ll = None, -np.inf
    for th in starts:
        th = _clip(th, d)
        m = np.zeros_like(th); v = np.zeros_like(th)
        for t in range(1, steps + 1):
            k = Kernel(np.exp(th[:d]), np.exp(th[d]), kind)
            try:
                ll, g = log_marginal_likelihood(X, ys, k, max(np.exp(th[d + 1]), noise_floor), grad=True)
            except np.linalg.LinAlgError:
                break
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g**2
            th = _clip(th + lr * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8), d)
        k = Kernel(np.exp(th[:d]), np.exp(th[d]), kind)
        sn2 = max(np.exp(th[d + 1]), noise_floor)
        try:
            ll = log_marginal_likelihood(X, ys, k, sn2)
        except np.linalg.LinAlgError:
            continue
        if ll > best_ll:
            best, best_ll = (k, sn2), ll
    k, sn2 = best
    gp = GP(k, sn2).fit(X, y)
    gp.log_marglik = best_ll
    return gp
