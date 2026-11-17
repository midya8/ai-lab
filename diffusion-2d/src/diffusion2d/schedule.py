"""Noise schedules for discrete-time DDPM.

Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, eps ~ N(0, I),
with abar_t = prod_{s<=t} (1 - beta_s), t = 1..T.  Index 0 of every array
below is the clean data point (abar_0 = 1), so arrays have length T + 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Schedule:
    betas: np.ndarray       # length T + 1, betas[0] = 0
    alphas: np.ndarray      # 1 - betas
    abar: np.ndarray        # cumulative product, abar[0] = 1

    @property
    def T(self) -> int:
        return len(self.betas) - 1

    def posterior_variance(self, t: int) -> float:
        """beta_tilde_t = (1 - abar_{t-1}) / (1 - abar_t) * beta_t (true reverse-step variance)."""
        return float((1.0 - self.abar[t - 1]) / (1.0 - self.abar[t]) * self.betas[t])


def _from_betas(b: np.ndarray) -> Schedule:
    betas = np.concatenate([[0.0], np.asarray(b, dtype=np.float64)])
    alphas = 1.0 - betas
    return Schedule(betas, alphas, np.cumprod(alphas))


def linear_schedule(T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02) -> Schedule:
    """Ho et al. linear betas, rescaled so the total noise does not depend on T
    (the published 1e-4..0.02 values are for T = 1000)."""
    scale = 1000.0 / T
    return _from_betas(np.linspace(beta_start * scale, beta_end * scale, T))


def cosine_schedule(T: int = 100, s: float = 0.008, max_beta: float = 0.999) -> Schedule:
    """Nichol & Dhariwal cosine schedule."""
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    abar = f / f[0]
    betas = np.clip(1.0 - abar[1:] / abar[:-1], 0.0, max_beta)
    return _from_betas(betas)


def make_schedule(name: str, T: int) -> Schedule:
    if name == "linear":
        return linear_schedule(T)
    if name == "cosine":
        return cosine_schedule(T)
    raise ValueError(f"unknown schedule {name!r}")
