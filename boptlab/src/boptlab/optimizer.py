"""Bayesian optimisation loop (minimisation) in the unit cube, with sequential and batch proposals."""
from __future__ import annotations
import numpy as np
from .gp import GP, Kernel, fit_gp
from . import acquisition as A

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]


def halton(n, d, skip=20):
    """First n points of the Halton sequence in [0,1]^d (low-discrepancy design / baseline)."""
    if d > len(_PRIMES):
        raise ValueError("dimension too large for the built-in Halton primes")
    out = np.empty((n, d))
    for j in range(d):
        b = _PRIMES[j]
        for i in range(n):
            f, r, k = 1.0, 0.0, i + 1 + skip
            while k > 0:
                f /= b; r += f * (k % b); k //= b
            out[i, j] = r
    return out


def random_search(f, bounds, n, rng, quasi=False):
    """Baseline: n evaluations of f at uniform (or Halton, with a random skip so seeds differ) points. Returns (X, y)."""
    bounds = np.asarray(bounds, float)
    U = halton(n, len(bounds), skip=int(rng.integers(0, 10000))) if quasi else rng.random((n, len(bounds)))
    X = bounds[:, 0] + U * (bounds[:, 1] - bounds[:, 0])
    return X, np.array([f(x) for x in X])


def optimize_acquisition(score, d, rng, anchors=None, n_random=2000, n_refine=4, n_local=40):
    """Maximise score(U) -> (m,) over [0,1]^d: random + Halton + near-anchor candidates, then shrinking-radius local search."""
    parts = [rng.random((n_random, d)), halton(max(n_random // 4, 16), d, skip=int(rng.integers(0, 1000)))]
    if anchors is not None and len(anchors):
        a = np.atleast_2d(anchors)
        for s in (0.02, 0.08):
            parts.append(np.clip(np.repeat(a, 100, 0) + s * rng.standard_normal((100 * len(a), d)), 0, 1))
    C = np.vstack(parts)
    s = score(C)
    order = np.argsort(-s)[:n_refine]
    best_x, best_s = C[order[0]].copy(), s[order[0]]
    for i in order:
        x, sx, rad = C[i].copy(), s[i], 0.1
        for _ in range(n_local):
            P = np.clip(x + rad * rng.standard_normal((16, d)), 0, 1)
            sp = score(P)
            j = int(np.argmax(sp))
            if sp[j] > sx:
                x, sx = P[j], sp[j]
            else:
                rad *= 0.7
            if rad < 1e-4:
                break
        if sx > best_s:
            best_x, best_s = x, sx
    return best_x


class BayesOpt:
    """GP-based Bayesian optimiser. acquisition in {'ei','pi','ucb','ts'}; set noisy=True for noisy objectives
    (incumbent becomes the best *posterior mean* at observed points instead of the best raw observation)."""

    def __init__(self, bounds, acquisition="ei", kernel="matern52", n_init=None, noisy=False, beta=2.0, xi=0.0,
                 rng=None, refit_every=1, fit_kwargs=None):
        if acquisition not in ("ei", "pi", "ucb", "ts"):
            raise ValueError("acquisition must be one of ei, pi, ucb, ts")
        self.bounds = np.asarray(bounds, float)
        self.d = len(self.bounds)
        self.acq, self.kernel_kind, self.noisy, self.beta, self.xi = acquisition, kernel, noisy, beta, xi
        self.n_init = n_init if n_init is not None else max(5, 2 * self.d)
        self.rng = rng or np.random.default_rng(0)
        self.refit_every, self.fit_kwargs = refit_every, fit_kwargs or {}
        self.U = np.empty((0, self.d)); self.y = np.empty(0)
        self.gp = None
        self._hyp, self._asks = None, 0
        self._init = halton(self.n_init, self.d, skip=int(self.rng.integers(0, 500)))

    # ----- coordinate maps
    def to_unit(self, X):
        return (np.atleast_2d(X) - self.bounds[:, 0]) / (self.bounds[:, 1] - self.bounds[:, 0])

    def from_unit(self, U):
        return self.bounds[:, 0] + np.atleast_2d(U) * (self.bounds[:, 1] - self.bounds[:, 0])

    @property
    def X(self):
        return self.from_unit(self.U)

    def tell(self, X, y):
        self.U = np.vstack([self.U, self.to_unit(X)])
        self.y = np.concatenate([self.y, np.ravel(y)])
        self.gp = None

    def _fit(self):
        self._asks += 1
        if self._hyp is None or self._asks % self.refit_every == 0:
            gp = fit_gp(self.U, self.y, self.kernel_kind, rng=self.rng, init=self._hyp, **self.fit_kwargs)
            self._hyp = (gp.kernel.ls.copy(), gp.kernel.sf2, gp.sn2)
        else:
            ls, sf2, sn2 = self._hyp
            gp = GP(Kernel(ls, sf2, self.kernel_kind), sn2).fit(self.U, self.y)
        self.gp = gp
        return gp

    def incumbent(self, gp):
        if not self.noisy:
            return float(self.y.min())
        return float(gp.predict(self.U)[0].min())

    def ask(self, q=1):
        """Propose q points in original coordinates. Initial space-filling design first, then acquisition-driven."""
        n_done = len(self.y)
        if n_done < self.n_init:
            k = min(q, self.n_init - n_done)
            U = self._init[n_done:n_done + k]
            return self.from_unit(U)
        gp = self._fit()
        best = self.incumbent(gp)
        anchors = self.U[np.argsort(self.y)[:3]]
        picks = []
        cur = gp
        for _ in range(q):
            if self.acq == "ts":
                C = np.vstack([self.rng.random((800, self.d)),
                               np.clip(np.repeat(anchors, 100, 0) + 0.05 * self.rng.standard_normal((100 * len(anchors), self.d)), 0, 1)])
                draw = cur.sample(C, 1, self.rng)[0]
                u = C[int(np.argmin(draw))]
            else:
                g = cur
                def score(C, g=g):
                    mu, sd = g.predict(C)
                    if self.acq == "ei":
                        return A.expected_improvement(mu, sd, best, self.xi)
                    if self.acq == "pi":
                        return A.probability_of_improvement(mu, sd, best, self.xi)
                    return A.confidence_bound(mu, sd, self.beta)
                u = optimize_acquisition(score, self.d, self.rng, anchors)
            picks.append(u)
            if q > 1:  # constant liar (CL-min): pretend the pick returned the current best value
                cur = cur.condition(u[None, :], [self.y.min()])
        return self.from_unit(np.array(picks))

    def run(self, f, n_total, q=1):
        """Evaluate f until n_total observations exist. Returns the history of best-so-far (raw) values."""
        while len(self.y) < n_total:
            X = self.ask(min(q, n_total - len(self.y)))
            self.tell(X, [f(x) for x in X])
        return np.minimum.accumulate(self.y)

    def best(self):
        i = int(np.argmin(self.y))
        return self.X[i], float(self.y[i])
