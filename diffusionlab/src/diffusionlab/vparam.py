"""v-prediction (Salimans & Ho 2022) conversions and SNR-based loss weights.

With alpha = sqrt(abar_t), sigma = sqrt(1 - abar_t) (alpha^2 + sigma^2 = 1) and
x_t = alpha x0 + sigma eps, the velocity target is v = alpha eps - sigma x0.  The
maps between parametrisations are exact:
    eps = sigma x_t + alpha v        x0 = alpha x_t - sigma v
so any eps-predictor (including the Bayes-optimal oracle) can be exposed as a
v-predictor and back, and sampling is unchanged up to float error.
Loss weights, expressed as a multiplier on the eps-MSE (SNR = alpha^2/sigma^2):
    eps-MSE: 1;  x0-MSE: SNR;  v-MSE: SNR + 1;  min-SNR-gamma (Hang et al.): min(SNR, gamma)
"""
from __future__ import annotations

import numpy as np

from .diffusion import EpsFn
from .schedule import Schedule


def _as(sched: Schedule, t: int):
    ab = sched.abar[t]
    return np.sqrt(ab), np.sqrt(1.0 - ab)


def v_target(x0: np.ndarray, eps: np.ndarray, t: int, sched: Schedule) -> np.ndarray:
    a, s = _as(sched, t)
    return a * eps - s * x0


def eps_from_v(xt: np.ndarray, v: np.ndarray, t: int, sched: Schedule) -> np.ndarray:
    a, s = _as(sched, t)
    return s * xt + a * v


def v_from_eps(xt: np.ndarray, eps: np.ndarray, t: int, sched: Schedule) -> np.ndarray:
    a, s = _as(sched, t)
    x0 = (xt - s * eps) / a
    return a * eps - s * x0


def eps_fn_to_v_fn(eps_fn: EpsFn, sched: Schedule) -> EpsFn:
    """Wrap an eps-predictor as a v-predictor v(x, t, labels)."""
    return lambda x, t, labels: v_from_eps(x, eps_fn(x, t, labels), t, sched)


def v_fn_to_eps_fn(v_fn: EpsFn, sched: Schedule) -> EpsFn:
    """Wrap a v-predictor so it can drive the eps-based samplers (DDIM, DPM-Solver-2)."""
    return lambda x, t, labels: eps_from_v(x, v_fn(x, t, labels), t, sched)


def snr(sched: Schedule) -> np.ndarray:
    """SNR_t = abar/(1-abar) for t = 1..T (index t-1 -> array position t; position 0 is unused, set inf)."""
    ab = sched.abar
    out = np.full(len(ab), np.inf)
    out[1:] = ab[1:] / (1.0 - ab[1:])
    return out


def loss_weights(sched: Schedule, kind: str, gamma: float = 5.0) -> np.ndarray:
    """Multiplier on the eps-MSE per timestep (array length T+1, entry 0 unused)."""
    r = snr(sched)
    if kind == "eps":
        w = np.ones_like(r)
    elif kind == "x0":
        w = r
    elif kind == "v":
        w = r + 1.0
    elif kind == "min_snr":
        w = np.minimum(r, gamma)
    else:
        raise ValueError(f"unknown weighting {kind!r}")
    return w
