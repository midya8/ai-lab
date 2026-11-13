"""Target-leakage screen: features that predict the label suspiciously well on their own."""
from __future__ import annotations

import numpy as np


def _binned_accuracy(x: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Held-out accuracy of a one-feature histogram classifier (quantile bins, majority label).

    Even rows fit the bin->label table, odd rows score it, so memorisation is not rewarded.
    """
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    if x.size < 4 or np.all(x == x[0]):
        return float("nan")
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    b = np.searchsorted(edges, x, side="right")
    fit, ev = np.arange(x.size) % 2 == 0, np.arange(x.size) % 2 == 1
    classes = np.unique(y)
    table = {}
    for k in np.unique(b[fit]):
        labs, cnt = np.unique(y[fit][b[fit] == k], return_counts=True)
        table[k] = labs[np.argmax(cnt)]
    prior = classes[np.argmax([(y[fit] == c).sum() for c in classes])]
    pred = np.array([table.get(k, prior) for k in b[ev]])
    return float((pred == y[ev]).mean())


def leaky_features(X, y, names=None, bins: int = 10, threshold: float = 0.95) -> list[dict]:
    """Rank features by single-feature held-out accuracy; flag those >= threshold.

    Returns dicts {feature, accuracy, baseline, flagged} sorted by accuracy. A flag means
    "check where this column comes from", not proof of leakage: some features are simply
    strong. Compare with `baseline` (majority-class rate).
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    if X.ndim != 2 or len(y) != len(X):
        raise ValueError("X must be 2-D with len(y) rows")
    names = names or [f"f{i}" for i in range(X.shape[1])]
    _, cnt = np.unique(y, return_counts=True)
    base = float(cnt.max() / len(y))
    out = []
    for j, nm in enumerate(names):
        a = _binned_accuracy(X[:, j], y, bins)
        if np.isnan(a):
            continue
        out.append({"feature": nm, "accuracy": a, "baseline": base, "flagged": a >= threshold})
    out.sort(key=lambda r: -r["accuracy"])
    return out
