"""Coverage diagnostics. Marginal coverage can hold while conditional coverage fails badly."""
from __future__ import annotations

import numpy as np


def interval_coverage(lo, hi, y):
    return float(np.mean((np.asarray(y) >= lo) & (np.asarray(y) <= hi)))


def mean_width(lo, hi):
    w = np.asarray(hi) - np.asarray(lo)
    return float(np.mean(w))


def binned_coverage(x, covered, bins=5):
    """Coverage within equal-count bins of a 1-D covariate x -> (bin_cov array, worst-bin gap)."""
    order = np.argsort(np.asarray(x).reshape(-1))
    chunks = np.array_split(np.asarray(covered, float)[order], bins)
    cov = np.array([c.mean() for c in chunks])
    return cov


def set_coverage(S, y):
    return float(np.mean(S[np.arange(len(S)), np.asarray(y, int)]))


def class_conditional_coverage(S, y, K):
    y = np.asarray(y, int)
    return np.array([S[y == k, k].mean() if np.any(y == k) else np.nan for k in range(K)])


def mean_set_size(S):
    return float(S.sum(1).mean())


def empty_rate(S):
    return float((S.sum(1) == 0).mean())


def coverage_beta_params(n, alpha):
    """Exact law of split-conformal coverage given n calibration points (continuous scores):
    Beta(k, n+1-k) with k = ceil((n+1)(1-alpha)). Returns (k, n+1-k)."""
    import math
    k = math.ceil((n + 1) * (1 - alpha))
    return k, n + 1 - k


def size_stratified_coverage(S, y):
    """Coverage within each prediction-set-size stratum (the SSC diagnostic of Angelopoulos et al.).

    Returns {size: (coverage, count)}. Marginal coverage can be met while the hardest (largest-set) inputs
    are badly under-covered; this exposes that. Returns the worst stratum via `worst_size_stratum`.
    """
    S = np.asarray(S, bool)
    y = np.asarray(y, int)
    sizes = S.sum(1)
    hit = S[np.arange(len(y)), y]
    return {int(s): (float(hit[sizes == s].mean()), int((sizes == s).sum())) for s in np.unique(sizes)}


def worst_size_stratum(S, y, min_count=30):
    """Lowest coverage among size strata with at least `min_count` points (avoids noise from tiny strata)."""
    strata = {s: v for s, v in size_stratified_coverage(S, y).items() if v[1] >= min_count}
    if not strata:
        raise ValueError("no stratum has enough points")
    s = min(strata, key=lambda k: strata[k][0])
    return s, strata[s][0]


def set_outcome_breakdown(S, y):
    """Fractions of points whose set is: empty, a correct singleton, a wrong singleton,
    or multi-label (containing / missing the truth). Fractions sum to 1."""
    S = np.asarray(S, bool)
    y = np.asarray(y, int)
    size = S.sum(1)
    hit = S[np.arange(len(S)), y]
    n = len(S)
    return {
        "empty": float(np.mean(size == 0)),
        "singleton_correct": float(np.mean((size == 1) & hit)),
        "singleton_wrong": float(np.mean((size == 1) & ~hit)),
        "multi_covered": float(np.mean((size > 1) & hit)),
        "multi_missed": float(np.mean((size > 1) & ~hit)),
    }


def worst_class_coverage(S, y, K, min_count=30):
    """(class, coverage) of the lowest-coverage class among classes with >= min_count points."""
    y = np.asarray(y, int)
    cc = class_conditional_coverage(S, y, K)
    counts = np.bincount(y, minlength=K)
    ok = [k for k in range(K) if counts[k] >= min_count]
    if not ok:
        raise ValueError("no class has enough points")
    k = min(ok, key=lambda j: cc[j])
    return int(k), float(cc[k])


def interval_score(lo, hi, y, alpha=0.1):
    """Mean Winkler interval score: width + (2/alpha) * distance outside the interval.

    A proper scoring rule for central (1-alpha) intervals; lower is better. It rewards narrow
    intervals but punishes misses, so it cannot be gamed by width alone.
    """
    lo, hi, y = (np.asarray(a, float) for a in (lo, hi, y))
    pen = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean((hi - lo) + pen))


def wilson_interval(covered, z=1.96):
    """Wilson score interval for an empirical coverage rate from 0/1 outcomes -> (lo, hi).

    Use it to tell a genuine coverage failure from test-set noise: with n test points the
    sampling sd of coverage is about sqrt(c(1-c)/n), so 0.88 on n=100 does not refute a 0.90 target.
    """
    c = np.asarray(covered, float).reshape(-1)
    n = len(c)
    if n == 0:
        raise ValueError("need at least one outcome")
    p = c.mean()
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def coverage_calibration_curve(mean_fn, X_cal, y_cal, X_test, y_test, alphas=(0.05, 0.1, 0.2, 0.3, 0.5), scale_fn=None):
    """Validity curve for split conformal: for each alpha, calibrate once on (X_cal, y_cal) and
    measure empirical coverage on the test set. Returns rows (nominal=1-alpha, empirical, gap).
    Under exchangeability every gap should be >= 0 up to test-set noise (and < 1/(n_cal+1) on average)."""
    from .conformal import SplitConformal
    rows = []
    for a in alphas:
        sc = SplitConformal(mean_fn, scale_fn, alpha=a).calibrate(X_cal, y_cal)
        lo, hi = sc.predict(X_test)
        emp = interval_coverage(lo, hi, y_test)
        rows.append((1 - a, float(emp), float(emp - (1 - a))))
    return rows


def selective_accuracy(S, y):
    """Abstain unless the set is a singleton; return (answer_rate, accuracy_on_answered).

    accuracy is nan when nothing is answered. Under split conformal the *singleton* error is not
    controlled at alpha (only marginal miscoverage is), so this is an empirical diagnostic.
    """
    S = np.asarray(S, bool)
    y = np.asarray(y, int)
    single = S.sum(1) == 1
    rate = float(single.mean())
    if not single.any():
        return rate, float("nan")
    return rate, float(S[single, y[single]].mean())
