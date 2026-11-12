"""Class-balance and train/test label-shift checks."""
from __future__ import annotations

import numpy as np


def class_balance(y: np.ndarray, rare_fraction: float = 0.05) -> dict:
    """Per-class counts, imbalance ratio (max/min count) and the classes below `rare_fraction`."""
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("empty label array")
    classes, counts = np.unique(y, return_counts=True)
    frac = counts / counts.sum()
    return {
        "counts": {int(c) if np.issubdtype(type(c), np.integer) else c: int(n) for c, n in zip(classes.tolist(), counts)},
        "fractions": {c: float(f) for c, f in zip(classes.tolist(), frac)},
        "imbalance_ratio": float(counts.max() / counts.min()),
        "rare_classes": [c for c, f in zip(classes.tolist(), frac) if f < rare_fraction],
    }


def label_shift(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Total-variation distance and chi-square statistic between train and test label distributions.

    The chi-square statistic uses the train distribution as the expectation; classes absent
    from train but present in test are reported in `unseen_in_train` (they make chi2 infinite).
    """
    y_train, y_test = np.asarray(y_train), np.asarray(y_test)
    classes = np.union1d(y_train, y_test)
    p = np.array([(y_train == c).mean() for c in classes])
    q = np.array([(y_test == c).mean() for c in classes])
    n = len(y_test)
    unseen = [c for c, pi, qi in zip(classes.tolist(), p, q) if pi == 0 and qi > 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, (q * n - p * n) ** 2 / (p * n), np.where(q > 0, np.inf, 0.0))
    return {
        "tv_distance": float(0.5 * np.abs(p - q).sum()),
        "chi2": float(terms.sum()),
        "dof": int((p > 0).sum() - 1),
        "unseen_in_train": unseen,
    }
