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
