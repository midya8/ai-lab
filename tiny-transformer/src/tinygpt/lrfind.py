"""Learning-rate range test (Smith 2017): sweep LR exponentially, record smoothed loss, suggest an LR.

The model's weights are restored afterwards, so the sweep has no side effects.
"""
from __future__ import annotations

import math

import numpy as np

from .optim import Adam


def lr_schedule(lr_min: float, lr_max: float, steps: int) -> list[float]:
    if not (0 < lr_min < lr_max) or steps < 2:
        raise ValueError("need 0 < lr_min < lr_max and steps >= 2")
    r = (lr_max / lr_min) ** (1 / (steps - 1))
    return [lr_min * r ** i for i in range(steps)]


def suggest_lr(lrs, losses, divide: float = 10.0) -> float:
    """LR at the steepest loss descent (min slope in log-LR), before divergence."""
    n = len(losses)
    if n < 3:
        raise ValueError("need at least 3 points")
    ll = np.log(np.asarray(lrs, dtype=float))
    ls = np.asarray(losses, dtype=float)
    slope = np.gradient(ls, ll)
    i = int(np.argmin(slope))
    return float(lrs[i])


def find_lr(model, data, batch, block, lr_min=1e-6, lr_max=1.0, steps=60, beta=0.9,
            diverge_factor=4.0, seed=0, get_batch=None):
    """Run the sweep; returns dict(lrs, losses, suggested, stopped_early)."""
    if get_batch is None:
        from .train import get_batch
    saved = {k: v.copy() for k, v in model.params.items()}
    opt = Adam(model.params, lr=lr_min)
    rng = np.random.default_rng(seed)
    lrs, losses = [], []
    avg, best = 0.0, math.inf
    stopped = False
    try:
        for i, lr in enumerate(lr_schedule(lr_min, lr_max, steps), 1):
            opt.lr = lr
            x, y = get_batch(data, batch, block, rng)
            loss, grads = model.loss_and_grads(x, y)
            if not math.isfinite(loss):
                stopped = True
                break
            avg = beta * avg + (1 - beta) * loss
            smooth = avg / (1 - beta ** i)  # bias-corrected EMA
            lrs.append(lr)
            losses.append(smooth)
            best = min(best, smooth)
            if i > 5 and smooth > diverge_factor * best:
                stopped = True
                break
            opt.step(grads)
    finally:
        for k, v in saved.items():
            model.params[k][...] = v
    # cut the diverged tail so it doesn't dominate the slope
    suggested = suggest_lr(lrs, losses) if len(lrs) >= 3 else lr_min
    return {"lrs": lrs, "losses": losses, "suggested": suggested, "stopped_early": stopped}
