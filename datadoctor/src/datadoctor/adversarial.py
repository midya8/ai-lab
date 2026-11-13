"""Adversarial validation: can a classifier tell train rows from test rows?

Marginal tests (KS/PSI) miss shifts that live only in feature *interactions*
(e.g. a changed correlation). A domain classifier trained to separate the two
splits catches them: out-of-fold AUC near 0.5 means indistinguishable, near 1
means strong shift. Linear model (SoftmaxRegression) + a quadratic-feature
option so correlation changes become linearly visible.
"""
from __future__ import annotations

import numpy as np

from .cv import stratified_kfold
from .models import SoftmaxRegression


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC AUC (ties get average rank)."""
    y, score = np.asarray(y), np.asarray(score, float)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("need both classes")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2 + 1
        i = j + 1
    r = np.empty(len(s))
    r[order] = ranks
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _quad(X: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    cols = [X] + [(X[:, i] * X[:, j])[:, None] for i in range(d) for j in range(i, d)]
    return np.hstack(cols)


def adversarial_validation(X_train, X_test, quadratic: bool = False, k: int = 5, seed: int = 0,
                           n_permutations: int = 20) -> dict:
    """Out-of-fold AUC of a train-vs-test classifier plus a permutation null.

    Returns auc, the null mean/max AUC over label permutations (same folds
    procedure), a p-value ((1+#null>=auc)/(1+n)) and per-feature |coef| ranking
    from a final fit (linear features only; indicative, not causal).
    """
    A, B = np.asarray(X_train, float), np.asarray(X_test, float)
    if A.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError("train and test must be 2-D with the same number of features")
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A), int), np.ones(len(B), int)]
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    F = _quad(Z) if quadratic else Z
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)

    def oof(labels):
        score = np.zeros(len(labels))
        for tr, te in stratified_kfold(labels, k, seed):
            m = SoftmaxRegression(seed=seed).fit(F[tr], labels[tr], 2)
            score[te] = m.predict_proba(F[te])[:, 1]
        return auc(labels, score)

    observed = oof(y)
    rng = np.random.default_rng(seed)
    null = np.array([oof(rng.permutation(y)) for _ in range(n_permutations)])
    p = float((1 + (null >= observed).sum()) / (1 + n_permutations))
    m = SoftmaxRegression(seed=seed).fit(Z, y, 2)
    w = np.abs(m.W[:, 1] - m.W[:, 0])
    return {"auc": observed, "null_mean": float(null.mean()), "null_max": float(null.max()),
            "p_value": p, "shifted": bool(p <= 0.05 and observed > 0.6),
            "top_features": [int(i) for i in np.argsort(-w)[:5]]}
