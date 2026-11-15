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


class MondrianSplitConformal:
    """Group-conditional split conformal for regression: one threshold per group (e.g. an x-bin), so
    coverage is >= 1-alpha *within every group* with enough calibration points. Groups with fewer
    than min_calibration_size(alpha) points get an infinite interval (the honest answer)."""

    def __init__(self, mean_fn, group_fn, scale_fn=None, alpha=0.1):
        self.mean_fn, self.group_fn, self.scale_fn, self.alpha = mean_fn, group_fn, scale_fn, alpha
        self.q = {}

    def _scale(self, X):
        return np.ones(len(X)) if self.scale_fn is None else np.maximum(self.scale_fn(X), 1e-6)

    def calibrate(self, X, y):
        g = np.asarray(self.group_fn(X))
        r = np.abs(np.asarray(y) - self.mean_fn(X)) / self._scale(X)
        self.q = {k: conformal_quantile(r[g == k], self.alpha) for k in np.unique(g)}
        return self

    def predict(self, X):
        g = np.asarray(self.group_fn(X))
        q = np.array([self.q.get(k, np.inf) for k in g])   # unseen group -> infinite interval
        m, h = self.mean_fn(X), q * self._scale(X)
        return m - h, m + h


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


def _raps_penalty(K, lam, k_reg):
    """Penalty for the label at 1-indexed rank j: lam * max(0, j - k_reg)."""
    return lam * np.maximum(0, np.arange(1, K + 1) - k_reg)


def raps_scores(P, y, u, lam=0.01, k_reg=1):
    """Regularised APS (Angelopoulos et al. 2021): randomised APS plus a rank penalty that discourages
    large sets built from many tiny-probability labels. lam=0 recovers aps_scores."""
    order = np.argsort(-P, axis=1)
    rank = np.argmax(order == np.asarray(y, int)[:, None], axis=1)
    return aps_scores(P, y, u) + _raps_penalty(P.shape[1], lam, k_reg)[rank]


def raps_sets(P, q, u, lam=0.01, k_reg=1):
    order = np.argsort(-P, axis=1)
    sp = np.take_along_axis(P, order, 1)
    cum = np.cumsum(sp, 1)
    sc = cum - sp + u[:, None] * sp + _raps_penalty(P.shape[1], lam, k_reg)[None, :]
    S = np.zeros(sc.shape, bool)
    np.put_along_axis(S, order, sc <= q, 1)
    return S


class ConformalClassifier:
    """Split-conformal prediction sets from a probability function.

    method: "lac" | "aps" | "raps" (raps: lam, k_reg; not combined with mondrian).   mondrian=True calibrates one threshold per true class, which buys
    class-conditional (not just marginal) coverage at the price of fewer calibration points per class.
    """

    def __init__(self, proba_fn, alpha=0.1, method="lac", mondrian=False, seed=0, lam=0.01, k_reg=1):
        if method not in ("lac", "aps", "raps"):
            raise ValueError(method)
        if method == "raps" and mondrian:
            raise ValueError("raps does not support mondrian")
        self.lam, self.k_reg = lam, k_reg
        self.proba_fn, self.alpha, self.method, self.mondrian = proba_fn, alpha, method, mondrian
        self.rng = np.random.default_rng(seed)
        self.q = None

    def _scores(self, P, y):
        if self.method == "lac":
            return lac_scores(P, y)
        u = self.rng.uniform(size=len(P))
        if self.method == "raps":
            return raps_scores(P, y, u, self.lam, self.k_reg)
        return aps_scores(P, y, u)

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
        if self.method == "raps":
            return raps_sets(P, self.q, self.rng.uniform(size=len(P)), self.lam, self.k_reg)
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


# ------------------------------------------------------------------ covariate shift
def weighted_conformal_quantile(scores, weights, w_test, alpha):
    """Tibshirani et al. (2019) weighted quantile for ONE test point: the (1-alpha)-quantile of
    sum_i p_i delta(s_i) + p_test delta(+inf), with p_i = w_i / (sum w + w_test).
    Returns +inf when the calibration mass cannot reach 1-alpha."""
    s = np.asarray(scores, float)
    w = np.asarray(weights, float)
    o = np.argsort(s)
    cum = np.cumsum(w[o]) / (w.sum() + w_test)
    j = np.searchsorted(cum, 1 - alpha - 1e-12)
    return np.inf if j >= len(s) else float(s[o][j])


class WeightedSplitConformal(SplitConformal):
    """SplitConformal under covariate shift. `weight_fn(X)` is the density ratio p_test(x)/p_cal(x)
    (known, or estimated e.g. by `logistic_density_ratio`); the interval width then depends on x."""

    def __init__(self, mean_fn, weight_fn, scale_fn=None, alpha=0.1):
        super().__init__(mean_fn, scale_fn, alpha)
        self.weight_fn = weight_fn

    def calibrate(self, X, y):
        self.cal_scores = np.abs(np.asarray(y) - self.mean_fn(X)) / self._scale(X)
        self.cal_w = np.asarray(self.weight_fn(X), float)
        return self

    def predict(self, X):
        wt = np.asarray(self.weight_fn(X), float)
        q = np.array([weighted_conformal_quantile(self.cal_scores, self.cal_w, w, self.alpha) for w in wt])
        m, h = self.mean_fn(X), q * self._scale(X)
        return m - h, m + h


def logistic_density_ratio(X_cal, X_test, degree=2, steps=400, lr=0.5, l2=1e-3):
    """Estimate p_test/p_cal with a polynomial logistic domain classifier (Newton-free GD):
    w(x) = (n_cal/n_test) * p(test|x) / p(cal|x). Returns a callable on X of shape (n, 1)."""
    def feats(X):
        x = np.asarray(X, float).reshape(len(X), -1)[:, :1]
        return np.hstack([x ** d for d in range(1, degree + 1)])
    F = np.vstack([feats(X_cal), feats(X_test)])
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = np.hstack([np.ones((len(F), 1)), (F - mu) / sd])
    t = np.r_[np.zeros(len(X_cal)), np.ones(len(X_test))]
    b = np.zeros(Z.shape[1])
    for _ in range(steps):
        p = 1 / (1 + np.exp(-Z @ b))
        b -= lr * (Z.T @ (p - t) / len(t) + l2 * b)
    ratio = len(X_cal) / len(X_test)

    def w(X):
        Zx = np.hstack([np.ones((len(X), 1)), (feats(X) - mu) / sd])
        return ratio * np.exp(np.clip(Zx @ b, -30, 30))
    return w


# ------------------------------------------------------------- jackknife+ / CV+ (no held-out split)
class JackknifePlus:
    """Jackknife+ / CV+ (Barber et al. 2021). fit_fn(X, y) -> predict callable.

    folds=None is leave-one-out (jackknife+); folds=K is K-fold CV+. Every point is used for both
    training and calibration; guaranteed coverage is >= 1 - 2*alpha (typically ~ 1 - alpha).
    Interval at x: [ q-_alpha{mu_{-i}(x) - R_i}, q+_alpha{mu_{-i}(x) + R_i} ], R_i the held-out residual
    of the model that did not see point i."""

    def __init__(self, fit_fn, alpha=0.1, folds=None, seed=0):
        self.fit_fn, self.alpha, self.folds, self.seed = fit_fn, alpha, folds, seed

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y, float)
        n = len(y)
        K = n if self.folds is None else self.folds
        fold = (np.arange(n) if self.folds is None
                else np.random.default_rng(self.seed).permutation(n) % K)
        self.models, self.fold_of = [], fold
        self.R = np.empty(n)
        for k in range(K):
            te = fold == k
            m = self.fit_fn(X[~te], y[~te])
            self.models.append(m)
            self.R[te] = np.abs(y[te] - m(X[te]))
        return self

    def predict(self, X):
        n = len(self.R)
        P = np.stack([m(X) for m in self.models])[self.fold_of]       # (n, len(X)): mu_{-k(i)}(x)
        k = math.ceil((n + 1) * (1 - self.alpha))
        if k > n:
            inf = np.full(P.shape[1], np.inf)
            return -inf, inf
        hi = np.sort(P + self.R[:, None], axis=0)[k - 1]
        lo = -np.sort(-(P - self.R[:, None]), axis=0)[k - 1]
        return lo, hi


def conformal_risk_control(loss_fn, lambdas, cal_data, alpha, B=1.0):
    """Conformal risk control (Angelopoulos et al. 2022) for losses that are
    non-increasing in lambda and bounded by B.

    loss_fn(lam, item) -> loss in [0, B]; lambdas ascending grid.
    Returns the smallest lambda with (n/(n+1)) * mean_loss + B/(n+1) <= alpha,
    so E[loss at the chosen lambda on a fresh exchangeable point] <= alpha
    (up to the grid). Returns the largest lambda if none qualifies, with
    `ok=False` in the second return value."""
    lambdas = list(lambdas)
    n = len(cal_data)
    if n == 0:
        raise ValueError("empty calibration set")
    for lam in lambdas:
        risk = float(np.mean([loss_fn(lam, d) for d in cal_data]))
        if n / (n + 1) * risk + B / (n + 1) <= alpha:
            return lam, True
    return lambdas[-1], False


# ------------------------------------------------------------ conformal predictive distributions
class ConformalPredictiveDistribution:
    """Split conformal predictive distribution (Vovk et al.): a full CDF for y given x, not only an interval.

    Signed normalised residuals r_i = (y_i - mean(x_i)) / scale(x_i) on a calibration set give
    F(y | x) = (#{r_i <= (y - mean(x)) / scale(x)} + U) / (n + 1), with U~Uniform(0,1) (randomised) or
    U=1/2 (deterministic midpoint). The randomised version's probability integral transform is exactly
    uniform marginally under exchangeability, so quantiles are calibrated at *every* level at once.
    """

    def __init__(self, mean_fn, scale_fn=None, randomised=False, seed=0):
        self.mean_fn, self.scale_fn, self.randomised = mean_fn, scale_fn, randomised
        self.rng = np.random.default_rng(seed)
        self.r = None

    def _scale(self, X):
        return np.ones(len(X)) if self.scale_fn is None else np.asarray(self.scale_fn(X), float)

    def calibrate(self, X, y):
        self.r = np.sort((np.asarray(y, float) - self.mean_fn(X)) / self._scale(X))
        return self

    def cdf(self, X, y):
        """F(y_i | x_i) for each row (y broadcast to len(X) if scalar)."""
        n = len(self.r)
        t = (np.broadcast_to(np.asarray(y, float), (len(X),)) - self.mean_fn(X)) / self._scale(X)
        k = np.searchsorted(self.r, t, side="right")
        u = self.rng.uniform(size=len(t)) if self.randomised else 0.5
        return (k + u) / (n + 1)

    def quantile(self, X, p):
        """Predictive p-quantile of y | x (p in (0,1)); +-inf beyond the calibration resolution."""
        n = len(self.r)
        k = math.ceil(p * (n + 1))
        if k < 1:
            z = -np.inf
        elif k > n:
            z = np.inf
        else:
            z = self.r[k - 1]
        return self.mean_fn(X) + self._scale(X) * z

    def interval(self, X, alpha=0.1):
        """Equal-tailed (asymmetric) interval from the predictive distribution."""
        return self.quantile(X, alpha / 2), self.quantile(X, 1 - alpha / 2)

    def crps(self, X, y, grid=200):
        """Mean CRPS estimated from the calibration-residual empirical distribution (exact for a step CDF)."""
        y = np.asarray(y, float)
        out = []
        m, s = self.mean_fn(X), self._scale(X)
        r = self.r
        n = len(r)
        for i in range(len(y)):
            z = r * s[i] + m[i]                       # atoms of the predictive distribution
            out.append(np.mean(np.abs(z - y[i])) - 0.5 * np.mean(np.abs(z[:, None] - z[None, :])))
        return float(np.mean(out))
