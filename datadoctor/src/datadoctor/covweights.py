"""Covariate-shift importance weights via a train-vs-test domain classifier.

w(x) = p_test(x)/p_train(x) = (n_train/n_test) * p(test|x)/p(train|x).
Weights are computed out-of-fold for train rows (so the classifier never scores
rows it was fit on), optionally clipped, and summarised by Kish effective
sample size. Assumes only p(x) shifts (p(y|x) fixed) and overlapping support.
"""
from __future__ import annotations

import numpy as np

from .cv import stratified_kfold
from .models import SoftmaxRegression


def covariate_shift_weights(X_train, X_test, clip: float | None = 20.0, k: int = 5,
                            seed: int = 0, quadratic: bool = False) -> dict:
    A, B = np.asarray(X_train, float), np.asarray(X_test, float)
    if A.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("train and test must be 2-D with the same number of features")
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if quadratic:
        d = Z.shape[1]
        Z = np.hstack([Z] + [(Z[:, i] * Z[:, j])[:, None] for i in range(d) for j in range(i, d)])
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    p = np.zeros(len(y))
    for tr, te in stratified_kfold(y, k, seed):
        m = SoftmaxRegression(seed=seed).fit(Z[tr], y[tr], 2)
        p[te] = m.predict_proba(Z[te])[:, 1]
    p = np.clip(p[: len(A)], 1e-6, 1 - 1e-6)
    w = (len(A) / len(B)) * p / (1 - p)
    if clip is not None:
        w = np.minimum(w, clip)
    w = w / w.mean()
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    return {"weights": w, "ess": ess, "ess_fraction": ess / len(w), "max_weight": float(w.max())}
