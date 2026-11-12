"""Per-feature covariate drift between a train and a test split (KS test and PSI)."""
from __future__ import annotations

import numpy as np


def ks_statistic(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and its asymptotic p-value."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        raise ValueError("empty sample")
    grid = np.concatenate([a, b])
    d = float(np.abs(np.searchsorted(a, grid, "right") / a.size - np.searchsorted(b, grid, "right") / b.size).max())
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-9:
        return d, 1.0
    j = np.arange(1, 101)
    p = 2.0 * float(np.sum((-1.0) ** (j - 1) * np.exp(-2.0 * j**2 * lam**2)))
    return d, float(min(1.0, max(0.0, p)))


def psi(a: np.ndarray, b: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of b against a, using quantile bins of a."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(a, edges)[0] / a.size + eps
    q = np.histogram(b, edges)[0] / b.size + eps
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(X_train, X_test, names=None, alpha: float = 0.01, psi_threshold: float = 0.25) -> dict:
    """Rank features by KS statistic; flag those with p < alpha (Bonferroni) or PSI above the threshold.

    PSI rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.
    """
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    if X_train.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train and test must be 2-D with the same number of features")
    d = X_train.shape[1]
    names = names or [f"f{i}" for i in range(d)]
    rows = []
    for j in range(d):
        ks, p = ks_statistic(X_train[:, j], X_test[:, j])
        s = psi(X_train[:, j], X_test[:, j])
        rows.append({"feature": names[j], "ks": ks, "p_value": p, "psi": s,
                     "flagged": bool(p < alpha / d or s > psi_threshold)})
    rows.sort(key=lambda r: -r["ks"])
    return {"features": rows, "flagged": [r["feature"] for r in rows if r["flagged"]]}
