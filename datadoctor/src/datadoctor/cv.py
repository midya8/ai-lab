"""Stratified k-fold and out-of-sample probabilities."""

from __future__ import annotations

from typing import Callable

import numpy as np


def stratified_kfold(y: np.ndarray, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``k`` (train_idx, test_idx) pairs; every index is in exactly one test fold.

    Each class is shuffled and dealt round-robin, so class proportions per fold differ
    by at most one example, and classes smaller than ``k`` still spread over folds.
    """
    y = np.asarray(y, dtype=int)
    if k < 2:
        raise ValueError("k must be >= 2")
    if len(y) < k:
        raise ValueError("fewer examples than folds")
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    offset = 0
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[(j + offset) % k].append(int(i))
        offset += len(idx)  # continue where the last class stopped so small classes do not pile on fold 0
    all_idx = np.arange(len(y))
    out = []
    for f in folds:
        te = np.array(sorted(f), dtype=int)
        mask = np.ones(len(y), dtype=bool)
        mask[te] = False
        out.append((all_idx[mask], te))
    return out


def cross_val_probs(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], object] | None = None,
    k: int = 5,
    seed: int = 0,
    repeats: int = 1,
    n_classes: int | None = None,
) -> np.ndarray:
    """Out-of-sample class probabilities: each row is predicted by a model that never saw it.

    ``model_factory()`` must return an object with ``fit(X, y, n_classes=)`` and
    ``predict_proba(X)``. With ``repeats > 1`` the probabilities of differently-split
    runs are averaged, which reduces the variance of a single split.
    """
    from .models import SoftmaxRegression

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    K = int(n_classes if n_classes is not None else y.max() + 1)
    factory = model_factory or SoftmaxRegression
    P = np.zeros((len(y), K))
    for r in range(repeats):
        for tr, te in stratified_kfold(y, k, seed + 1000 * r):
            m = factory()
            m.fit(X[tr], y[tr], n_classes=K)
            P[te] += m.predict_proba(X[te])
    return P / repeats
