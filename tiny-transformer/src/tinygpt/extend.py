"""Context-window extension for RoPE models: position interpolation and NTK-aware scaling.

A RoPE model trained on windows of `L` tokens has only ever seen rotation angles up to `L`
times each pair's frequency, and past that its attention degrades quickly. Two remedies change
how positions turn into angles instead of touching weights:

* ``linear``: position interpolation (Chen et al. 2023) divides every position by
  ``new_len / L`` so the longer window is squeezed back into the angle range the weights know.
* ``ntk``: NTK-aware scaling enlarges the frequency base, stretching low-frequency pairs by the
  full factor while leaving high-frequency (local) pairs nearly untouched.

``extend_context`` returns a *view*: a new GPT that shares the parameter arrays with the original
(nothing is copied, so fine-tuning the view fine-tunes the weights) but has a longer
``block_size`` and the scaling baked into its config, so evaluate / sliding-window eval /
generate / KV-cache decoding all work unchanged. It is saved and reloaded like any model.
"""
from __future__ import annotations

from dataclasses import replace

from .model import GPT


def extend_context(model: GPT, new_len: int, mode: str = "linear") -> GPT:
    """Return a view of `model` that attends over `new_len` tokens.

    `mode` is "linear", "ntk", or "none" (plain extrapolation: longer window, original angles,
    kept as the baseline to compare against). Scaling compounds with any scale already present,
    and is always measured against the length the weights were trained at.
    """
    cfg = model.cfg
    if not cfg.rope:
        raise ValueError("context extension needs a RoPE model (learned positions have no entries past block_size)")
    if new_len < 1:
        raise ValueError("new_len must be positive")
    if mode not in ("linear", "ntk", "none"):
        raise ValueError(f"unknown mode {mode!r}; use linear, ntk or none")
    trained = cfg.train_len or cfg.block_size
    scale = 1.0 if mode == "none" else max(new_len / trained, 1.0)
    new_cfg = replace(cfg, block_size=new_len, rope_scale=scale,
                      rope_scaling="linear" if mode == "none" else mode, train_len=trained)
    view = GPT.__new__(GPT)
    view.__dict__.update(model.__dict__)  # shares params (same arrays), flags like label_smoothing
    view.cfg = new_cfg
    return view


def with_window(model: GPT, window: int) -> GPT:
    """Same weights and same rope scaling, different evaluation window.

    Use it to score an extended model on short contexts: the angle schedule stays the one the
    model was extended (and possibly fine-tuned) with, only the window length changes.
    """
    if window < 1:
        raise ValueError("window must be positive")
    view = GPT.__new__(GPT)
    view.__dict__.update(model.__dict__)
    view.cfg = replace(model.cfg, block_size=window)
    return view


def finetune_extended(view: GPT, data, steps: int, batch: int = 8, lr: float = 1e-3, seed: int = 0) -> list:
    """Briefly train `view` on windows of its (extended) block_size; returns the loss per step.

    Position interpolation is meant to be followed by a short fine-tune at the new length: the
    weights adapt to the compressed angles far faster than they would learn a new length from
    scratch. Updates the shared parameters in place.
    """
    import numpy as np

    from .optim import Adam
    from .train import get_batch

    data = np.asarray(data)
    T = view.cfg.block_size
    if len(data) < T + 2:
        raise ValueError("data shorter than the extended window")
    opt = Adam(view.params, lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(steps):
        x, y = get_batch(data, batch, T, rng)
        loss, grads = view.loss_and_grads(x, y)
        opt.step(grads)
        losses.append(float(loss))
    return losses
