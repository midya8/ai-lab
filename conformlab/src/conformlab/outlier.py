"""Conformal outlier detection: finite-sample valid p-values and BH FDR control."""
import numpy as np


def conformal_pvalues(cal_scores, test_scores):
    """p_j = (1 + #{cal >= s_j}) / (n + 1); larger score = more outlying.
    Super-uniform under exchangeability with inliers (P(p <= a) <= a)."""
    cal = np.sort(np.asarray(cal_scores, float))
    s = np.asarray(test_scores, float)
    ge = len(cal) - np.searchsorted(cal, s, side="left")
    return (1.0 + ge) / (len(cal) + 1.0)


def benjamini_hochberg(p, q=0.1):
    """Boolean rejection mask controlling FDR <= q (valid for independent / PRDS p-values)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ok = p[order] <= q * np.arange(1, m + 1) / m
    mask = np.zeros(m, bool)
    if ok.any():
        k = np.max(np.nonzero(ok)[0])
        mask[order[: k + 1]] = True
    return mask


def conformal_outliers(cal_scores, test_scores, q=0.1):
    """Flag outliers in a test batch with FDR <= q * (fraction of inliers)."""
    p = conformal_pvalues(cal_scores, test_scores)
    return benjamini_hochberg(p, q), p
