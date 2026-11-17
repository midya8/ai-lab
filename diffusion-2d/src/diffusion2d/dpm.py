"""DPM-Solver-2 (midpoint, eps-prediction; Lu et al. 2022) on the discrete schedule.

Works in half-log-SNR lambda = log(alpha/sigma), alpha=sqrt(abar), sigma=sqrt(1-abar).
Exact one-step update from lambda_t to lambda_s (h = lambda_s - lambda_t > 0):
    x_s = (alpha_s/alpha_t) x_t - sigma_s (e^h - 1) eps
Midpoint variant uses eps evaluated at the intermediate point lambda_t + h/2; the
midpoint time is a continuous quantity, so abar is linearly interpolated in log space
and the eps function is called at the nearest integer timestep (a discretisation
approximation, tested to be no worse than DDIM at equal step count on the oracle).
The last step (to t=0, sigma=0) falls back to a first-order step (denoise to x0).
Cost: 2 network evaluations per step.
"""
from __future__ import annotations

import numpy as np

from .diffusion import EpsFn, timestep_grid
from .schedule import Schedule


def _lam(ab: float) -> float:
    return 0.5 * (np.log(ab) - np.log1p(-ab))


def sample_dpm2(eps_fn: EpsFn, sched: Schedule, n: int, D: int, steps: int,
                rng=None, labels=None, x_T=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal((n, D)) if x_T is None else x_T.copy()
    grid = timestep_grid(sched.T, steps)
    ts = np.arange(sched.T + 1)
    lam_tab = np.array([_lam(a) for a in sched.abar[1:]])        # t = 1..T, increasing as t decreases? decreasing in t
    for t, s in zip(grid[:-1], grid[1:]):
        ab_t, ab_s = sched.abar[t], sched.abar[s]
        a_t, sg_t = np.sqrt(ab_t), np.sqrt(1 - ab_t)
        eps = eps_fn(x, int(t), labels)
        if s == 0:
            x = (x - sg_t * eps) / a_t
            continue
        a_s, sg_s = np.sqrt(ab_s), np.sqrt(1 - ab_s)
        l_t, l_s = _lam(ab_t), _lam(ab_s)
        h = l_s - l_t
        l_m = l_t + h / 2
        # invert lambda -> timestep (lam_tab is decreasing in t)
        tm = int(np.clip(round(float(np.interp(-l_m, -lam_tab, ts[1:]))), s, t))
        ab_m = 1.0 / (1.0 + np.exp(-2 * l_m))                    # abar from lambda
        a_m, sg_m = np.sqrt(ab_m), np.sqrt(1 - ab_m)
        u = (a_m / a_t) * x - sg_m * np.expm1(h / 2) * eps
        eps_m = eps_fn(u, tm, labels)
        x = (a_s / a_t) * x - sg_s * np.expm1(h) * eps_m
    return x
