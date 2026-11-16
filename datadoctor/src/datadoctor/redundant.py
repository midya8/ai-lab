"""Redundant feature detection: duplicate or (anti-)collinear numeric columns.

Pairs with |Pearson r| >= threshold are reported; constant columns are skipped
(correlation undefined). Exact duplicates have r == 1 and identical values.
"""
from __future__ import annotations

import numpy as np


def redundant_features(X, names=None, threshold: float = 0.98) -> list[dict]:
    X = np.asarray(X, float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    n, d = X.shape
    names = list(names) if names is not None else [f"x{j}" for j in range(d)]
    if len(names) != d:
        raise ValueError("names length must match columns")
    if n < 2:
        return []
    Z = X - X.mean(0)
    sd = np.sqrt((Z ** 2).sum(0))
    ok = sd > 1e-12
    out = []
    for i in range(d):
        if not ok[i]:
            continue
        for j in range(i + 1, d):
            if not ok[j]:
                continue
            r = float(Z[:, i] @ Z[:, j] / (sd[i] * sd[j]))
            if abs(r) >= threshold:
                out.append({"a": names[i], "b": names[j], "r": r,
                            "exact": bool(np.array_equal(X[:, i], X[:, j]))})
    return sorted(out, key=lambda p: -abs(p["r"]))


def drop_redundant(X, names=None, threshold: float = 0.98):
    """Return indices of columns to keep (first of each redundant group is kept)."""
    X = np.asarray(X, float)
    names = list(names) if names is not None else [f"x{j}" for j in range(X.shape[1])]
    idx = {nm: k for k, nm in enumerate(names)}
    drop = set()
    for p in redundant_features(X, names, threshold):
        a, b = idx[p["a"]], idx[p["b"]]
        if a not in drop:
            drop.add(b)
    return [k for k in range(X.shape[1]) if k not in drop]
