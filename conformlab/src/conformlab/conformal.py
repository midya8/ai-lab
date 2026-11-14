"""Split-conformal predictors: regression (abs-residual, normalised, CQR) and classification
(LAC, randomised APS, Mondrian class-conditional), plus online adaptive conformal (ACI)."""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score; +inf when n is too small to be valid."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(s[k - 1])


def min_calibration_size(alpha):
    """Smallest n for which a finite interval is possible: ceil(1/alpha) - 1."""
    return math.ceil(1 / alpha) - 1


# ----------------------------------------------------------------------------- regression
class SplitConformal:
    """Interval = mean(x) +- q * scale(x). scale=None gives the classic constant-width interval."""

    def __init__(self, mean_fn, scale_fn=None, alpha=0.1):
        self.mean_fn, self.scale_fn, self.alpha, self.q = mean_fn, scale_fn, alpha, None

    def _scale(self, X):
        return np.ones(len(X)) if self.scale_fn is None else np.maximum(self.scale_fn(X), 1e-6)

    def calibrate(self, X, y):
        self.q = conformal_quantile(np.abs(np.asarray(y) - self.mean_fn(X)) / self._scale(X), self.alpha)
        return self

    def predict(self, X):
        m, h = self.mean_fn(X), self.q * self._scale(X)
        return m - h, m + h


class CQR:
    """Conformalised quantile regression (Romano et al. 2019): widen/narrow [lo, hi] by a constant."""

    def __init__(self, lo_fn, hi_fn, alpha=0.1):
        self.lo_fn, self.hi_fn, self.alpha, self.q = lo_fn, hi_fn, alpha, None

    def calibrate(self, X, y):
        y = np.asarray(y)
        self.q = conformal_quantile(np.maximum(self.lo_fn(X) - y, y - self.hi_fn(X)), self.alpha)
        return self

    def predict(self, X):
        return self.lo_fn(X) - self.q, self.hi_fn(X) + self.q


# ------------------------------------------------------------------------- classification
def lac_scores(P, y):
    """Least-ambiguous-set score: 1 - p(true class)."""
    return 1.0 - P[np.arange(len(P)), np.asarray(y, int)]


def aps_scores(P, y, u):
    """Randomised APS: mass of classes ranked above the label plus u * p(label), u ~ U(0,1)."""
    order = np.argsort(-P, axis=1)
    sp = np.take_along_axis(P, order, 1)
    cum = np.cumsum(sp, 1)
    rank = np.argmax(order == np.asarray(y, int)[:, None], axis=1)
    i = np.arange(len(P))
    return cum[i, rank] - sp[i, rank] + u * sp[i, rank]


def aps_sets(P, q, u):
    """Boolean (n, K) set matrix: include label k iff its randomised APS score <= q."""
    order = np.argsort(-P, axis=1)
    sp = np.take_along_axis(P, order, 1)
    cum = np.cumsum(sp, 1)
    keep_sorted = (cum - sp + u[:, None] * sp) <= q
    S = np.zeros_like(keep_sorted)
    np.put_along_axis(S, order, keep_sorted, 1)
    return S


class ConformalClassifier:
    """Split-conformal prediction sets from a probability function.

    method: "lac" | "aps".   mondrian=True calibrates one threshold per true class, which buys
    class-conditional (not just marginal) coverage at the price of fewer calibration points per class.
    """

    def __init__(self, proba_fn, alpha=0.1, method="lac", mondrian=False, seed=0):
        if method not in ("lac", "aps"):
            raise ValueError(method)
        self.proba_fn, self.alpha, self.method, self.mondrian = proba_fn, alpha, method, mondrian
        self.rng = np.random.default_rng(seed)
        self.q = None

    def _scores(self, P, y):
        if self.method == "lac":
            return lac_scores(P, y)
        return aps_scores(P, y, self.rng.uniform(size=len(P)))

    def calibrate(self, X, y):
        P, y = self.proba_fn(X), np.asarray(y, int)
        s = self._scores(P, y)
        if self.mondrian:
            self.q = np.array([conformal_quantile(s[y == k], self.alpha) for k in range(P.shape[1])])
        else:
            self.q = conformal_quantile(s, self.alpha)
        return self

    def predict_sets(self, X):
        P = self.proba_fn(X)
        if self.method == "lac":
            thr = (1.0 - self.q) if np.ndim(self.q) == 0 else (1.0 - self.q)[None, :]
            return P >= thr
        if self.mondrian:
            # per-class thresholds: label k is in iff its own randomised score <= q_k
            u = self.rng.uniform(size=len(P))
            order = np.argsort(-P, axis=1)
            sp = np.take_along_axis(P, order, 1)
            cum = np.cumsum(sp, 1)
            sc_sorted = cum - sp + u[:, None] * sp
            S = np.zeros(P.shape, bool)
            np.put_along_axis(S, order, sc_sorted <= self.q[order], 1)
            return S
        return aps_sets(P, self.q, self.rng.uniform(size=len(P)))


# ------------------------------------------------------------------------------- online
def aci(scores_fn, stream_len, alpha=0.1, gamma=0.02, window=None):
    """Adaptive conformal inference (Gibbs & Candes 2021) for a stream of residual scores.

    scores_fn(t) -> (calibration_scores_array, new_score). Level update:
        alpha_{t+1} = alpha_t + gamma * (alpha - err_t),   err_t = 1[score_t > q_t].
    Returns (miss indicators, alpha_t trajectory). gamma=0 is plain (non-adaptive) split conformal.
    """
    a_t, miss, traj = alpha, np.zeros(stream_len, bool), np.zeros(stream_len)
    for t in range(stream_len):
        cal, s = scores_fn(t)
        traj[t] = a_t
        if a_t <= 0:
            q = np.inf
        elif a_t >= 1:
            q = -np.inf
        else:
            q = conformal_quantile(cal, a_t)
        miss[t] = s > q
        a_t = a_t + gamma * (alpha - float(miss[t]))
    return miss, traj
