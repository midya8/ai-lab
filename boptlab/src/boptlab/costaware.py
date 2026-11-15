"""Cost-aware Bayesian optimisation: EI per unit cost under a *budget* (EIpu, Snoek et al. 2012).

Evaluations have a position-dependent cost c(x) > 0 (e.g. training time grows with model size). A second GP models
log c(x); the score is EI(x) / exp(mean log-cost), and the loop stops when the cumulative cost would exceed the budget."""
from __future__ import annotations
import numpy as np
from .gp import fit_gp
from . import acquisition as A
from .optimizer import halton, optimize_acquisition


class CostAwareBayesOpt:
    def __init__(self, bounds, budget, n_init=None, rng=None, fit_kwargs=None, cost_aware=True):
        self.bounds = np.asarray(bounds, float)
        self.d = len(self.bounds)
        self.budget = float(budget)
        self.n_init = n_init if n_init is not None else max(4, 2 * self.d)
        self.rng = rng or np.random.default_rng(0)
        self.fit_kwargs = fit_kwargs or {}
        self.cost_aware = cost_aware
        self.U = np.empty((0, self.d)); self.y = np.empty(0); self.c = np.empty(0)
        self._init = halton(self.n_init, self.d, skip=int(self.rng.integers(0, 500)))

    def from_unit(self, U):
        return self.bounds[:, 0] + np.atleast_2d(U) * (self.bounds[:, 1] - self.bounds[:, 0])

    @property
    def spent(self):
        return float(self.c.sum())

    def tell(self, x, y, cost):
        if cost <= 0:
            raise ValueError("cost must be positive")
        u = (np.asarray(x, float) - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])
        self.U = np.vstack([self.U, u]); self.y = np.append(self.y, y); self.c = np.append(self.c, cost)

    def ask(self):
        n = len(self.y)
        if n < self.n_init:
            return self.from_unit(self._init[n])
        gf = fit_gp(self.U, self.y, rng=self.rng, **self.fit_kwargs)
        gc = fit_gp(self.U, np.log(self.c), rng=self.rng, **self.fit_kwargs) if self.cost_aware else None
        best = float(self.y.min())

        def score(P):
            ei = A.expected_improvement(*gf.predict(P), best)
            return ei / np.exp(gc.predict(P)[0]) if gc is not None else ei

        anchors = self.U[np.argsort(self.y)[:3]]
        return self.from_unit(optimize_acquisition(score, self.d, self.rng, anchors))

    def run(self, f, cost_fn, max_evals=500):
        """Evaluate until the next *initial/proposed* point's cost would not fit. Returns (spent, best) trace arrays."""
        while len(self.y) < max_evals:
            x = self.ask()[0]
            c = float(cost_fn(x))
            if self.spent + c > self.budget:
                break
            self.tell(x, f(x), c)
        return self.trace()

    def trace(self):
        """(cumulative cost, best-so-far) per evaluation."""
        return np.cumsum(self.c), np.minimum.accumulate(self.y)

    def best(self):
        i = int(np.argmin(self.y))
        return self.from_unit(self.U[i])[0], float(self.y[i])


def best_at_budget(trace, b):
    """Best value found with cumulative cost <= b (inf if nothing fit)."""
    spent, best = trace
    ok = spent <= b
    return float(best[ok][-1]) if ok.any() else float("inf")
