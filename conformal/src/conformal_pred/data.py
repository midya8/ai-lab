"""Synthetic generators with known ground truth (so coverage claims can be checked exactly)."""
from __future__ import annotations

import numpy as np


def sigma_true(x, base=0.1, grow=0.6, width=3.0):
    """Noise scale of the heteroscedastic task: small near 0, large at the edges."""
    x = np.asarray(x, float).reshape(-1)
    return base + grow * (x / width) ** 2


def mean_true(x):
    x = np.asarray(x, float).reshape(-1)
    return np.sin(2 * x) + 0.3 * x


def hetero_regression(n, rng, noise="gauss", x_dist="uniform"):
    """1-D heteroscedastic regression. y = f(x) + sigma(x) * eps.

    noise: "gauss" or "t3" (Student-t, 3 dof, rescaled to unit variance: heavy tails).
    x_dist: "uniform" on [-3, 3] or "normal" N(0, 1).
    Returns X (n, 1) and y (n,).
    """
    x = rng.uniform(-3, 3, n) if x_dist == "uniform" else rng.normal(0, 1, n)
    if noise == "gauss":
        eps = rng.normal(size=n)
    elif noise == "t3":
        eps = rng.standard_t(3, size=n) / np.sqrt(3.0)
    else:
        raise ValueError(noise)
    y = mean_true(x) + sigma_true(x) * eps
    return x[:, None], y


def gaussian_classes(n, rng, K=6, d=4, spread=2.0, hard=(4, 5), hard_scale=2.5, priors=None):
    """K Gaussian classes in R^d. Classes in `hard` have a larger std (and sit closer together),
    so they are intrinsically harder: a classifier is less confident there, which is exactly what
    separates marginal from class-conditional coverage.

    The class means come from a fixed seed, so every call describes the same distribution.
    """
    mu = np.random.default_rng(12345).normal(size=(K, d)) * spread
    sd = np.ones(K)
    for k in hard:
        sd[k] = hard_scale
        mu[k] = mu[k] * 0.4
    p = np.full(K, 1.0 / K) if priors is None else np.asarray(priors, float) / np.sum(priors)
    y = rng.choice(K, size=n, p=p)
    X = mu[y] + rng.normal(size=(n, d)) * sd[y][:, None]
    return X, y
