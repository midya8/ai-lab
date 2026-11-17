"""State-space (Kalman) GP: exact O(n) inference for 1-D Matern-3/2 and Matern-5/2.

A Matern-(p+1/2) process is the first component of a linear SDE, so GP regression becomes
Kalman filtering plus an RTS smoother. The log marginal likelihood and posterior match
ExactGP (O(n^3)) to numerical precision while cost grows linearly in n.
"""
from __future__ import annotations

import numpy as np


def _expm(M: np.ndarray) -> np.ndarray:
    n = max(0, int(np.ceil(np.log2(max(np.abs(M).sum(axis=1).max(), 1e-12)))) + 1)
    A = M / (2.0 ** n)
    E = np.eye(M.shape[0])
    term = np.eye(M.shape[0])
    for k in range(1, 20):
        term = term @ A / k
        E = E + term
    for _ in range(n):
        E = E @ E
    return E


class KalmanMatern:
    def __init__(self, nu: float = 1.5, lengthscale: float = 1.0, variance: float = 1.0, noise: float = 0.1):
        if nu not in (1.5, 2.5):
            raise ValueError("nu must be 1.5 or 2.5")
        self.nu, self.ls, self.var, self.noise = nu, lengthscale, variance, noise

    def _model(self):
        s2, l = self.var, self.ls
        if self.nu == 1.5:
            lam = np.sqrt(3.0) / l
            F = np.array([[0.0, 1.0], [-lam**2, -2 * lam]])
            P = np.diag([s2, lam**2 * s2])
        else:
            lam = np.sqrt(5.0) / l
            F = np.array([[0, 1, 0], [0, 0, 1], [-lam**3, -3 * lam**2, -3 * lam]], dtype=float)
            k = lam**2 * s2 / 3.0
            P = np.array([[s2, 0, -k], [0, k, 0], [-k, 0, lam**4 * s2]])
        return F, P

    def _transition(self, dt: float):
        F, P = self._model()
        A = _expm(F * dt)
        Q = P - A @ P @ A.T
        return A, 0.5 * (Q + Q.T)

    def _run(self, t, y):
        """Filter over times t (y NaN = no observation). Returns lml and filter/pred moments."""
        F, P0 = self._model()
        d = F.shape[0]
        n = len(t)
        m, P = np.zeros(d), P0.copy()
        ms, Ps, mp, Pp, As = [], [], [], [], []
        lml = 0.0
        for i in range(n):
            if i == 0:
                A = np.eye(d)
                m_, P_ = m, P
            else:
                A, Q = self._transition(t[i] - t[i - 1])
                m_, P_ = A @ m, A @ P @ A.T + Q
            mp.append(m_); Pp.append(P_); As.append(A)
            if np.isnan(y[i]):
                m, P = m_, P_
            else:
                S = P_[0, 0] + self.noise
                K = P_[:, 0] / S
                r = y[i] - m_[0]
                lml += -0.5 * (np.log(2 * np.pi * S) + r * r / S)
                m = m_ + K * r
                P = P_ - np.outer(K, P_[0, :])
                P = 0.5 * (P + P.T)
            ms.append(m); Ps.append(P)
        return lml, ms, Ps, mp, Pp, As

    def log_marginal_likelihood(self, x, y) -> float:
        order = np.argsort(x)
        return self._run(np.asarray(x, float).ravel()[order], np.asarray(y, float)[order])[0]

    def predict(self, x, y, xs, include_noise: bool = False):
        """RTS-smoothed posterior mean/variance of f at xs (any order, may interleave training x)."""
        x = np.asarray(x, float).ravel(); xs = np.asarray(xs, float).ravel(); y = np.asarray(y, float)
        t = np.concatenate([x, xs])
        yy = np.concatenate([y, np.full(len(xs), np.nan)])
        order = np.argsort(t, kind="stable")
        ts, ys = t[order], yy[order]
        _, ms, Ps, mp, Pp, As = self._run(ts, ys)
        n = len(ts)
        sm, sP = [None] * n, [None] * n
        sm[-1], sP[-1] = ms[-1], Ps[-1]
        for i in range(n - 2, -1, -1):
            G = Ps[i] @ As[i + 1].T @ np.linalg.inv(Pp[i + 1])
            sm[i] = ms[i] + G @ (sm[i + 1] - mp[i + 1])
            sP[i] = Ps[i] + G @ (sP[i + 1] - Pp[i + 1]) @ G.T
        mean = np.array([s[0] for s in sm]); var = np.array([p[0, 0] for p in sP])
        inv = np.empty(n, int); inv[order] = np.arange(n)
        sel = inv[len(x):]
        v = var[sel] + (self.noise if include_noise else 0.0)
        return mean[sel], v
