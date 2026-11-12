"""Feature health checks: missing values, constant columns and redundant (highly correlated) pairs."""
from __future__ import annotations

import numpy as np


def feature_health(X, names=None, corr_threshold: float = 0.98) -> dict:
    """Report NaN fraction, constant features and pairs with |Pearson r| >= corr_threshold.

    Correlation uses pairwise-complete rows; constant columns are excluded from pairs.
    """
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = X.shape
    names = names or [f"f{i}" for i in range(d)]
    nan_frac = np.isnan(X).mean(axis=0) if n else np.zeros(d)
    constant = []
    for j in range(d):
        col = X[:, j][~np.isnan(X[:, j])]
        if col.size == 0 or np.all(col == col[0]):
            constant.append(names[j])
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            if names[i] in constant or names[j] in constant:
                continue
            ok = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            if ok.sum() < 3:
                continue
            a, b = X[ok, i], X[ok, j]
            if a.std() == 0 or b.std() == 0:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if abs(r) >= corr_threshold:
                pairs.append({"a": names[i], "b": names[j], "r": r})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {
        "missing": {names[j]: float(nan_frac[j]) for j in range(d) if nan_frac[j] > 0},
        "constant": constant,
        "redundant_pairs": pairs,
    }
