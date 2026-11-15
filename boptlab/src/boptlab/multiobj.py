"""Multi-objective Bayesian optimisation (minimisation) via ParEGO (Knowles, 2006):
each iteration draws random weights, scalarises the normalised objectives with the augmented Chebyshev function,
and runs one single-objective BO step on that scalar. Plus Pareto-front and exact 2-D hypervolume utilities."""
from __future__ import annotations
import numpy as np
from .optimizer import BayesOpt


def pareto_mask(Y):
    """Boolean mask of non-dominated rows of Y (all objectives minimised). Duplicates are all kept."""
    Y = np.asarray(Y, float)
    n = len(Y)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = (Y <= Y[i]).all(1) & (Y < Y[i]).any(1)
        if dom.any():
            keep[i] = False
    return keep


def hypervolume_2d(Y, ref):
    """Exact area dominated by the points Y (m,2) and bounded by the reference point ref (minimisation)."""
    Y = np.asarray(Y, float)
    ref = np.asarray(ref, float)
    Y = Y[(Y < ref).all(1)]
    if not len(Y):
        return 0.0
    Y = Y[pareto_mask(Y)]
    Y = Y[np.argsort(Y[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in Y:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return float(hv)


def chebyshev(Yn, w, rho=0.05):
    """Augmented Chebyshev scalarisation of normalised objectives Yn (n,k) with weights w (k,)."""
    wy = Yn * w
    return wy.max(1) + rho * wy.sum(1)


class ParEGO:
    def __init__(self, bounds, n_obj=2, n_init=None, rho=0.05, rng=None, **bo_kwargs):
        self.bounds = np.asarray(bounds, float)
        self.k, self.rho = n_obj, rho
        self.rng = rng or np.random.default_rng(0)
        self.n_init = n_init or max(6, 3 * len(self.bounds))
        self.bo_kwargs = bo_kwargs
        self._bo = BayesOpt(self.bounds, n_init=self.n_init, rng=self.rng, **bo_kwargs)
        self.X = np.empty((0, len(self.bounds)))
        self.Y = np.empty((0, n_obj))

    def tell(self, X, Y):
        X, Y = np.atleast_2d(X), np.atleast_2d(Y)
        if Y.shape[1] != self.k:
            raise ValueError("expected %d objectives per point" % self.k)
        self.X, self.Y = np.vstack([self.X, X]), np.vstack([self.Y, Y])

    def ask(self):
        n = len(self.Y)
        if n < self.n_init:
            return self._bo._init[n:n + 1] * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        lo, hi = self.Y.min(0), self.Y.max(0)
        Yn = (self.Y - lo) / np.where(hi > lo, hi - lo, 1.0)
        w = self.rng.dirichlet(np.ones(self.k))
        bo = BayesOpt(self.bounds, n_init=0, rng=self.rng, **self.bo_kwargs)
        bo.tell(self.X, chebyshev(Yn, w, self.rho))
        return bo.ask(1)

    def run(self, f, n_total):
        """f(x) -> sequence of k objective values. Returns the hypervolume trace if k == 2 (ref = running nadir x1.1)."""
        while len(self.Y) < n_total:
            x = self.ask()[0]
            self.tell(x, np.asarray(f(x), float))
        return self.front()

    def front(self):
        m = pareto_mask(self.Y)
        o = np.argsort(self.Y[m][:, 0])
        return self.X[m][o], self.Y[m][o]
