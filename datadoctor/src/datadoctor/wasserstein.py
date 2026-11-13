"""Scale-free 1-D Wasserstein-1 drift: unlike KS it measures *how far* mass moved, not just where CDFs differ most."""
from __future__ import annotations

import numpy as np


def wasserstein1(a, b) -> float:
    """Exact W1 distance between two empirical 1-D samples (integral of |F_a - F_b|)."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    if a.size == 0 or b.size == 0:
        raise ValueError("empty sample")
    grid = np.concatenate([a, b])
    grid.sort()
    fa = np.searchsorted(a, grid[:-1], "right") / a.size
    fb = np.searchsorted(b, grid[:-1], "right") / b.size
    return float(np.sum(np.abs(fa - fb) * np.diff(grid)))


def wasserstein_drift(X_train, X_test, names=None, threshold: float = 0.2) -> dict:
    """Per-feature W1 divided by the train standard deviation (a shift of 0.2 means 0.2 train-sigmas).

    Constant train features use scale 1. Features are flagged when the normalised distance exceeds `threshold`.
    The threshold is a rule of thumb, not calibrated against a null.
    """
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    if X_train.ndim != 2 or X_train.shape[1] != X_test.shape[1]:
        raise ValueError("train and test must be 2-D with the same number of features")
    d = X_train.shape[1]
    names = names or [f"f{i}" for i in range(d)]
    rows = []
    for j in range(d):
        s = float(X_train[:, j].std())
        s = s if s > 1e-12 else 1.0
        w = wasserstein1(X_train[:, j], X_test[:, j]) / s
        rows.append({"feature": names[j], "w1_norm": w, "flagged": bool(w > threshold)})
    rows.sort(key=lambda r: -r["w1_norm"])
    return {"features": rows, "flagged": [r["feature"] for r in rows if r["flagged"]]}
