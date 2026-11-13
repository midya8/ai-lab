"""Black-box label-shift estimation (BBSE, Lipton et al. 2018).

Given a classifier's predictions on labelled validation data and on unlabelled
test data, estimate the importance weights w[y] = p_test(y) / p_train(y)
assuming only the label prior shifted (p(x|y) fixed).
Solves C w = mu, C[i, j] = P(pred=i, y=j) on validation, mu[i] = P(pred=i) on test.
"""
from __future__ import annotations

import numpy as np


def bbse_weights(y_val, pred_val, pred_test, n_classes: int | None = None) -> dict:
    y_val, pred_val, pred_test = (np.asarray(a, int) for a in (y_val, pred_val, pred_test))
    if y_val.shape != pred_val.shape or y_val.size == 0 or pred_test.size == 0:
        raise ValueError("need equal-length non-empty validation labels/predictions and test predictions")
    k = n_classes or int(max(y_val.max(), pred_val.max(), pred_test.max()) + 1)
    C = np.zeros((k, k))
    np.add.at(C, (pred_val, y_val), 1.0)
    C /= y_val.size
    mu = np.bincount(pred_test, minlength=k) / pred_test.size
    if np.linalg.matrix_rank(C) < k:
        raise ValueError("confusion matrix is singular: some class is never separable by the classifier")
    w = np.clip(np.linalg.solve(C, mu), 0.0, None)
    prior = np.bincount(y_val, minlength=k) / y_val.size
    # renormalise so the implied test prior sums to 1
    implied = w * prior
    if implied.sum() > 0:
        w = w / implied.sum()
    return {"weights": w, "train_prior": prior, "test_prior": w * prior}
