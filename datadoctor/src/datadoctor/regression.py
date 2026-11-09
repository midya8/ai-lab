"""Label-error detection for regression targets.

Classification uses confident learning; for a continuous target the analogue is the
out-of-fold residual. Each row is predicted by a model that never saw it, the residuals
are scaled by a local robust spread (MAD within quantile bins of the prediction, so
heteroscedastic data is not unfairly flagged in its noisy region), and rows with a large
robust z-score are reported.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class RidgeRegression:
    """Closed-form ridge on standardised inputs (intercept unpenalised)."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegression":
        X = np.asarray(X, dtype=float)
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        Z = (X - self.mu) / self.sd
        self.y0 = float(np.mean(y))
        A = Z.T @ Z + self.l2 * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (np.asarray(y, dtype=float) - self.y0))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((np.asarray(X, dtype=float) - self.mu) / self.sd) @ self.w + self.y0


def cross_val_predict(X, y, model_factory: Callable[[], object] | None = None, k: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions; every row predicted by a model trained without it."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if k < 2 or n < k:
        raise ValueError("need k >= 2 and at least k rows")
    idx = np.random.default_rng(seed).permutation(n)
    pred = np.zeros(n)
    for f in range(k):
        te = idx[f::k]
        tr = np.setdiff1d(idx, te)
        m = (model_factory or RidgeRegression)()
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def robust_residual_scores(pred: np.ndarray, y: np.ndarray, bins: int = 5) -> np.ndarray:
    """|residual| / (1.4826 * local MAD), with the MAD taken inside prediction-quantile bins."""
    pred = np.asarray(pred, dtype=float)
    r = np.asarray(y, dtype=float) - pred
    n = len(r)
    bins = max(1, min(bins, n // 20))
    order = np.argsort(pred)
    scores = np.zeros(n)
    for chunk in np.array_split(order, bins):
        rc = r[chunk]
        mad = 1.4826 * np.median(np.abs(rc - np.median(rc)))
        scores[chunk] = np.abs(rc - np.median(rc)) / max(mad, 1e-12)
    return scores


def find_target_issues(X, y, threshold: float = 3.5, k: int = 5, seed: int = 0, model_factory=None, bins: int = 5):
    """Return ``(issue_indices_ranked, scores, out_of_fold_predictions)``."""
    pred = cross_val_predict(X, y, model_factory, k, seed)
    scores = robust_residual_scores(pred, y, bins)
    issues = np.flatnonzero(scores > threshold)
    return issues[np.argsort(-scores[issues])], scores, pred


def inject_target_noise(y, rate: float, scale: float = 4.0, seed: int = 0):
    """Corrupt ``rate`` of targets by +/- ``scale`` * std(y) (random sign). Returns (noisy, mask)."""
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    m = int(round(rate * len(y)))
    mask = np.zeros(len(y), dtype=bool)
    mask[rng.choice(len(y), size=m, replace=False)] = True
    out = y.copy()
    out[mask] += rng.choice([-1.0, 1.0], size=m) * scale * y.std() * rng.uniform(0.5, 1.0, size=m)
    return out, mask
