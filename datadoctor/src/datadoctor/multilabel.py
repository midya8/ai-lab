"""Multi-label issue detection: confident learning applied per class (one-vs-rest).

Each of the K labels is an independent binary problem. For label k we treat the
column ``[1 - P[:, k], P[:, k]]`` as 2-class probabilities and run the standard
confident-learning pipeline, so a missing tag (given 0, model confident of 1) and a
wrong tag (given 1, model confident of 0) are both found. Per-label results are
returned as an (n x K) boolean mask plus a ranked list of (example, label) cells.
"""

from __future__ import annotations

import numpy as np

from .confident import find_label_issues, label_quality_scores
from .models import SoftmaxRegression


def cross_val_probs_multilabel(X, Y, model_factory=None, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(label k = 1) for every example, one binary model per label."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=int)
    n, K = Y.shape
    model_factory = model_factory or (lambda: SoftmaxRegression(epochs=200))
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(n) % folds
    P = np.zeros((n, K))
    for k in range(K):
        for f in range(folds):
            tr, te = fold_of != f, fold_of == f
            yk = Y[tr, k]
            if yk.min() == yk.max():  # a fold with a constant label: use its rate
                P[te, k] = float(yk[0])
                continue
            m = model_factory().fit(X[tr], yk, n_classes=2)
            P[te, k] = m.predict_proba(X[te])[:, 1]
    return P


def find_multilabel_issues(P: np.ndarray, Y: np.ndarray, method: str = "both") -> np.ndarray:
    """Boolean (n x K) mask: True where the given tag looks wrong (missing or spurious)."""
    P = np.asarray(P, dtype=float)
    Y = np.asarray(Y, dtype=int)
    if P.shape != Y.shape:
        raise ValueError(f"P {P.shape} and Y {Y.shape} must have the same shape")
    if not set(np.unique(Y)) <= {0, 1}:
        raise ValueError("Y must be a 0/1 indicator matrix")
    out = np.zeros(Y.shape, dtype=bool)
    for k in range(Y.shape[1]):
        yk = Y[:, k]
        if yk.min() == yk.max():  # a label that is constant can't be audited
            continue
        P2 = np.stack([1 - P[:, k], P[:, k]], axis=1)
        out[:, k] = find_label_issues(P2, yk, method=method)
    return out


def rank_multilabel_issues(P: np.ndarray, Y: np.ndarray, mask: np.ndarray | None = None, top: int | None = None):
    """(example, label, kind, quality) cells, worst first. kind is 'missing' (given 0) or 'spurious' (given 1)."""
    P = np.asarray(P, dtype=float)
    Y = np.asarray(Y, dtype=int)
    cells = []
    for k in range(Y.shape[1]):
        P2 = np.stack([1 - P[:, k], P[:, k]], axis=1)
        q = label_quality_scores(P2, Y[:, k], "margin")
        idx = np.flatnonzero(mask[:, k]) if mask is not None else np.arange(len(Y))
        for i in idx:
            cells.append((int(i), k, "missing" if Y[i, k] == 0 else "spurious", float(q[i])))
    cells.sort(key=lambda c: c[3])
    return cells[:top] if top else cells


def inject_multilabel_noise(Y: np.ndarray, rate: float, seed: int = 0):
    """Flip ``rate`` of all cells (drops and additions alike). Returns (noisy, flipped_mask)."""
    Y = np.asarray(Y, dtype=int)
    rng = np.random.default_rng(seed)
    m = int(round(rate * Y.size))
    flat = np.zeros(Y.size, dtype=bool)
    flat[rng.choice(Y.size, size=m, replace=False)] = True
    flip = flat.reshape(Y.shape)
    return np.where(flip, 1 - Y, Y), flip


def make_multilabel_blobs(n: int = 1500, n_labels: int = 6, dim: int = 12, seed: int = 0):
    """Each label is a noisy linear function of X, so labels are learnable and co-occur."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim))
    W = rng.normal(size=(dim, n_labels)) * 1.2
    logits = X @ W + rng.normal(scale=0.3, size=(n, n_labels))
    return X, (logits > 0.4).astype(int)


def score_multilabel(flagged: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((flagged & truth).sum())
    prec = tp / max(int(flagged.sum()), 1)
    rec = tp / max(int(truth.sum()), 1)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "flagged": int(flagged.sum()), "true": int(truth.sum())}
