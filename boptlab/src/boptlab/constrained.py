"""Constrained Bayesian optimisation: minimise f(x) subject to c_k(x) <= 0 (Gardner et al. 2014 / Gelbart et al. 2014).

Each of f and the c_k gets its own GP. The score is EI over the best *feasible* observation times the product of the
per-constraint probabilities of feasibility, P(c_k(x) <= 0) = Phi(-mu_k / sd_k). Until a feasible point is seen, the
score is the probability of feasibility alone (find any feasible region first)."""
from __future__ import annotations
import numpy as np
from .gp import fit_gp
from . import acquisition as A
from .optimizer import halton, optimize_acquisition


def prob_feasible(mu, sd):
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    ok = sd > 1e-12
    return np.where(ok, A.norm_cdf(-mu / np.where(ok, sd, 1.0)), (mu <= 0).astype(float))


class ConstrainedBayesOpt:
    def __init__(self, bounds, n_constraints, n_init=None, rng=None, fit_kwargs=None):
        self.bounds = np.asarray(bounds, float)
        self.d, self.m = len(self.bounds), n_constraints
        self.n_init = n_init if n_init is not None else max(6, 3 * self.d)
        self.rng = rng or np.random.default_rng(0)
        self.fit_kwargs = fit_kwargs or {}
        self.U = np.empty((0, self.d)); self.y = np.empty(0); self.C = np.empty((0, self.m))
        self._init = halton(self.n_init, self.d, skip=int(self.rng.integers(0, 500)))

    def from_unit(self, U):
        return self.bounds[:, 0] + np.atleast_2d(U) * (self.bounds[:, 1] - self.bounds[:, 0])

    @property
    def X(self):
        return self.from_unit(self.U)

    def feasible_mask(self):
        return (self.C <= 0).all(1)

    def tell(self, X, y, C):
        X = np.atleast_2d(X)
        self.U = np.vstack([self.U, (X - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])])
        self.y = np.concatenate([self.y, np.ravel(y)])
        self.C = np.vstack([self.C, np.reshape(C, (len(X), self.m))])

    def ask(self):
        n = len(self.y)
        if n < self.n_init:
            return self.from_unit(self._init[n])
        gf = fit_gp(self.U, self.y, rng=self.rng, **self.fit_kwargs)
        gcs = [fit_gp(self.U, self.C[:, k], rng=self.rng, **self.fit_kwargs) for k in range(self.m)]
        feas = self.feasible_mask()
        best = float(self.y[feas].min()) if feas.any() else None

        def score(P):
            pf = np.ones(len(P))
            for g in gcs:
                pf = pf * prob_feasible(*g.predict(P))
            if best is None:
                return pf
            return A.expected_improvement(*gf.predict(P), best) * pf

        if best is not None:
            anchors = self.U[feas][np.argsort(self.y[feas])[:3]]
        else:
            anchors = self.U[np.argsort(np.maximum(self.C, 0).sum(1))[:3]]
        return self.from_unit(optimize_acquisition(score, self.d, self.rng, anchors))

    def run(self, f, constraints, n_total):
        """f(x) -> float; constraints(x) -> sequence of m values (feasible iff all <= 0). Returns best-feasible-so-far
        trace (nan before the first feasible observation)."""
        while len(self.y) < n_total:
            x = self.ask()[0]
            self.tell(x, [f(x)], [list(constraints(x))])
        return self.trace()

    def trace(self):
        vals = np.where(self.feasible_mask(), self.y, np.inf)
        t = np.minimum.accumulate(vals)
        return np.where(np.isinf(t), np.nan, t)

    def best(self):
        feas = self.feasible_mask()
        if not feas.any():
            return None
        i = np.flatnonzero(feas)[int(np.argmin(self.y[feas]))]
        return self.X[i], float(self.y[i])
