"""Synthetic regression problems with known generating processes."""
from __future__ import annotations

import numpy as np

from .gp import ExactGP
from .kernels import Kernel


def sample_gp_data(kernel: Kernel, n: int, noise: float, seed: int = 0, low: float = -3.0,
                   high: float = 3.0, dim: int = 1):
    """Draw X ~ U[low, high]^dim and y from the GP prior plus Gaussian noise.
    Returns X, y_noisy, f_clean."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, (n, dim))
    g = ExactGP(kernel, noise)
    g.set_data(X, np.zeros(n))
    f = g.sample_prior(X, 1, seed=seed + 1, include_noise=False)[0]
    y = f + np.sqrt(noise) * rng.standard_normal(n)
    return X, y, f


def smooth_function(x: np.ndarray) -> np.ndarray:
    """A fixed non-GP test function with several scales."""
    x = np.asarray(x).ravel()
    return np.sin(1.5 * x) + 0.4 * np.sin(5.0 * x) + 0.2 * x


def seasonal_trend(n: int, noise: float = 0.15, seed: int = 0, t_max: float = 12.0):
    """Linear trend + period-1 seasonality whose amplitude drifts slowly, on t in [0, t_max]."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, t_max, n))
    amp = 1.0 + 0.3 * np.sin(2 * np.pi * t / (2.5 * t_max))
    f = 0.35 * t + amp * np.sin(2 * np.pi * t)
    return t[:, None], f + np.sqrt(noise) * rng.standard_normal(n), f
