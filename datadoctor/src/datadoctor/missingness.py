"""Informative-missingness screen: does a feature being NaN predict the label?"""
from __future__ import annotations

import numpy as np


def _chi2_2xk(miss: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    classes = np.unique(y)
    obs = np.array([[np.sum((miss == m) & (y == c)) for c in classes] for m in (0, 1)], float)
    row, col, n = obs.sum(1, keepdims=True), obs.sum(0, keepdims=True), obs.sum()
    exp = row @ col / n
    stat = float(((obs - exp) ** 2 / exp).sum())
    return stat, len(classes) - 1


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square(k) via the regularised upper incomplete gamma (series/continued fraction)."""
    import math

    if k <= 0:
        return 1.0
    a, x2 = k / 2.0, x / 2.0
    if x2 <= 0:
        return 1.0
    if x2 < a + 1:
        term = s = 1.0 / a
        n = a
        for _ in range(500):
            n += 1
            term *= x2 / n
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return float(max(0.0, 1.0 - s * math.exp(-x2 + a * math.log(x2) - math.lgamma(a))))
    tiny = 1e-300
    b = x2 + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return float(min(1.0, math.exp(-x2 + a * math.log(x2) - math.lgamma(a)) * h))


def informative_missingness(X, y, names=None, alpha: float = 0.01, min_missing: int = 5) -> list[dict]:
    """Chi-square test of the NaN indicator of each feature against the label.

    Returns features with >= min_missing NaNs (and >= min_missing non-NaNs) whose
    p-value is below alpha / n_tested (Bonferroni), sorted by p. A hit means the
    missingness itself carries label information (a leak risk or a collection
    artefact); it does not say why.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    if X.ndim != 2 or len(y) != len(X):
        raise ValueError("X must be 2-D and match y")
    n, d = X.shape
    names = names or [f"f{i}" for i in range(d)]
    cand = []
    for j in range(d):
        miss = np.isnan(X[:, j]).astype(int)
        if miss.sum() < min_missing or (n - miss.sum()) < min_missing:
            continue
        stat, dof = _chi2_2xk(miss, y)
        cand.append({"feature": names[j], "chi2": stat, "p": _chi2_sf(stat, dof),
                     "missing_rate_by_class": {str(c): float(miss[y == c].mean()) for c in np.unique(y)}})
    thr = alpha / max(len(cand), 1)
    out = [c for c in cand if c["p"] < thr]
    out.sort(key=lambda c: c["p"])
    return out
