"""Gradient accumulation: emulate a large batch with several small micro-batches.

Loss is a mean over tokens, so averaging equal-sized micro-batch gradients
equals the full-batch gradient exactly (up to float error).
"""
from __future__ import annotations

import numpy as np


def accumulate_grads(model, x, y, micro_batch, rng=None):
    """Return (mean_loss, mean_grads) over x/y split into micro-batches of `micro_batch` rows."""
    B = len(x)
    if micro_batch <= 0 or B % micro_batch:
        raise ValueError(f"batch {B} must be divisible by micro_batch {micro_batch}")
    n = B // micro_batch
    total, loss_sum = None, 0.0
    for i in range(n):
        sl = slice(i * micro_batch, (i + 1) * micro_batch)
        loss, g = model.loss_and_grads(x[sl], y[sl], rng=rng)
        loss_sum += float(loss)
        if total is None:
            total = {k: v.astype(np.float64).copy() for k, v in g.items()}
        else:
            for k, v in g.items():
                total[k] += v
    return loss_sum / n, {k: (v / n).astype(model.params[k].dtype) for k, v in total.items()}
