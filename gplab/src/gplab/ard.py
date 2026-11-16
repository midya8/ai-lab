"""Automatic relevance determination: learn per-dimension lengthscales and rank features."""
from __future__ import annotations

import numpy as np

from .gp import ExactGP
from .kernels import RBF


def fit_ard(X, y, restarts: int = 2, seed: int = 0) -> ExactGP:
    """Fit an ARD-RBF exact GP by marginal likelihood."""
    X = np.asarray(X, float)
    gp = ExactGP(RBF(lengthscale=1.0, variance=1.0, ard_dims=X.shape[1]), noise=0.1, center=True)
    gp.fit(X, np.asarray(y, float), restarts=restarts, seed=seed)
    return gp


def relevance(gp: ExactGP) -> np.ndarray:
    """Relevance = 1 / lengthscale, normalised to sum to 1 (inputs should be standardised)."""
    r = 1.0 / np.asarray(gp.kernel.lengthscale, float)
    return r / r.sum()


def rank_features(X, y, **kw) -> np.ndarray:
    """Feature indices from most to least relevant."""
    Xs = (np.asarray(X, float) - np.mean(X, 0)) / (np.std(X, 0) + 1e-12)
    return np.argsort(-relevance(fit_ard(Xs, y, **kw)))
