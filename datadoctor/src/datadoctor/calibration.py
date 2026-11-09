"""Temperature scaling applied to out-of-sample probabilities before confident learning.

Confident learning's per-class thresholds assume probabilities mean something. An
over-confident model (sharp softmax) or an under-confident one (flat softmax, e.g. a
heavily regularised model or kNN with small k) shifts thresholds and the estimated noise
matrix. Fitting a single temperature T on the *given* labels (log P / T, renormalised)
minimises NLL. The labels are noisy, so this under-sharpens slightly; that is a known
trade-off, and the ranking of examples inside one class is unchanged (T is monotone).
"""

from __future__ import annotations

import numpy as np


def apply_temperature(P: np.ndarray, T: float) -> np.ndarray:
    """Return softmax(log P / T). T > 1 flattens, T < 1 sharpens."""
    if T <= 0:
        raise ValueError("temperature must be positive")
    L = np.log(np.clip(np.asarray(P, dtype=float), 1e-12, 1.0)) / T
    L -= L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def nll(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def expected_calibration_error(P: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = P.max(axis=1)
    hit = (P.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(ece)


def fit_temperature(P: np.ndarray, y: np.ndarray, lo: float = 0.05, hi: float = 20.0, iters: int = 60) -> float:
    """Golden-section search for the T minimising NLL (log-spaced; NLL is unimodal in log T)."""
    y = np.asarray(y, dtype=int)
    f = lambda lt: nll(apply_temperature(P, float(np.exp(lt))), y)  # noqa: E731
    a, b = np.log(lo), np.log(hi)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return float(np.exp((a + b) / 2))


def calibrate(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (calibrated probabilities, fitted temperature)."""
    T = fit_temperature(P, y)
    return apply_temperature(P, T), T
