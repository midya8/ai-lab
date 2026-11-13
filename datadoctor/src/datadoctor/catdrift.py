"""Drift for categorical features: chi-square test of homogeneity, PSI and unseen-category rate."""
from __future__ import annotations

import math

import numpy as np


def _chi2_sf(x: float, k: int) -> float:
    """Survival function of chi-square with k dof (regularised upper incomplete gamma, series/CF)."""
    if x <= 0 or k <= 0:
        return 1.0
    a, x2 = k / 2.0, x / 2.0
    if x2 < a + 1:
        term = s = 1.0 / a
        n = a
        for _ in range(1000):
            n += 1
            term *= x2 / n
            s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return float(max(0.0, 1.0 - s * math.exp(-x2 + a * math.log(x2) - math.lgamma(a))))
    b = x2 + 1 - a
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = 1.0 / (an * d + b) if an * d + b else 1e300
        c = b + an / c
        h *= d * c
        if abs(d * c - 1) < 1e-14:
            break
    return float(min(1.0, h * math.exp(-x2 + a * math.log(x2) - math.lgamma(a))))


def categorical_drift(train, test, eps: float = 1e-4, alpha: float = 0.01, psi_threshold: float = 0.25) -> dict:
    """Compare category frequencies of `test` against `train` (1-D sequences of hashables)."""
    train, test = list(train), list(test)
    if not train or not test:
        raise ValueError("empty sample")
    cats = sorted(set(train) | set(test), key=str)
    ct = np.array([sum(1 for v in train if v == c) for c in cats], float)
    cs = np.array([sum(1 for v in test if v == c) for c in cats], float)
    p, q = ct / ct.sum() + eps, cs / cs.sum() + eps
    psi = float(np.sum((q - p) * np.log(q / p)))
    tot = ct + cs
    n1, n2, n = ct.sum(), cs.sum(), ct.sum() + cs.sum()
    chi2 = float(np.sum((ct - tot * n1 / n) ** 2 / (tot * n1 / n) + (cs - tot * n2 / n) ** 2 / (tot * n2 / n)))
    dof = len(cats) - 1
    pval = _chi2_sf(chi2, dof) if dof > 0 else 1.0
    seen = set(train)
    unseen = sum(1 for v in test if v not in seen) / len(test)
    return {"chi2": chi2, "dof": dof, "p_value": pval, "psi": psi, "unseen_rate": unseen,
            "unseen_categories": [c for c in cats if c not in seen],
            "flagged": bool(pval < alpha or psi > psi_threshold or unseen > 0)}
