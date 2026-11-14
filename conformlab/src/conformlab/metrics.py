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
