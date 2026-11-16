"""Adam training loop with streaming data, cosine LR decay and dead-latent resampling."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import FeatureWorld
from .sae import SAE


@dataclass
class FitResult:
    losses: list = field(default_factory=list)
    resampled: int = 0


def resample_dead(sae: SAE, x: np.ndarray, dead: np.ndarray, rng: np.random.Generator,
                  opt_state: dict | None = None) -> int:
    """Re-initialise dead latents toward inputs the SAE currently reconstructs worst.

    Each dead latent gets a decoder row equal to a (normalised) high-error residual, an encoder
    column pointing the same way (scaled small), and a zero bias. Returns how many were reset.
    """
    idx_dead = np.where(dead)[0]
    if len(idx_dead) == 0:
        return 0
    xh = sae.reconstruct(x)
    resid = x - xh
    err = (resid ** 2).sum(axis=1)
    if err.sum() <= 0:
        return 0
    prob = err ** 2
    prob /= prob.sum()
    pick = rng.choice(len(x), size=len(idx_dead), p=prob)
    new = resid[pick]
    new = new / np.maximum(np.linalg.norm(new, axis=1, keepdims=True), 1e-12)
    p = sae.params
    p["W_dec"][idx_dead] = new
    p["W_enc"][:, idx_dead] = new.T * 0.2
    p["b_enc"][idx_dead] = 0.0
    if opt_state is not None:
        for key in ("m", "v"):
            opt_state[key]["W_dec"][idx_dead] = 0.0
            opt_state[key]["W_enc"][:, idx_dead] = 0.0
            opt_state[key]["b_enc"][idx_dead] = 0.0
    return int(len(idx_dead))


def fit(sae: SAE, world: FeatureWorld, steps: int = 3000, batch: int = 512, lr: float = 2e-3,
        seed: int = 0, resample_every: int = 0, resample_until: float = 0.6,
        log_every: int = 0, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> FitResult:
    """Train on fresh samples from ``world`` each step (infinite data, no overfitting).

    ``b_dec`` starts at the data mean. If ``resample_every > 0``, latents that never fired in the
    last window are re-initialised every that many steps until ``resample_until * steps``.
    """
    rng = np.random.default_rng(seed + 1000)
    x0, _ = world.sample(4096, rng)
    sae.params["b_dec"][:] = x0.mean(axis=0)
    m = {k: np.zeros_like(v) for k, v in sae.params.items()}
    v = {k: np.zeros_like(x) for k, x in sae.params.items()}
    state = {"m": m, "v": v}
    res = FitResult()
    fired = np.zeros(sae.m, dtype=bool)
    for t in range(1, steps + 1):
        x, _ = world.sample(batch, rng)
        loss, grads, info = sae.loss_and_grads(x)
        fired |= (info["a"] > 0).any(axis=0)
        sae.project_decoder_grad(grads)
        cur = lr * 0.5 * (1 + np.cos(np.pi * (t - 1) / steps)) + lr * 0.02
        for k, p in sae.params.items():
            g = grads[k]
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v[k] = beta2 * v[k] + (1 - beta2) * g * g
            mh = m[k] / (1 - beta1 ** t)
            vh = v[k] / (1 - beta2 ** t)
            p -= cur * mh / (np.sqrt(vh) + eps)
        state["m"], state["v"] = m, v
        sae.normalize_decoder()
        if log_every and t % log_every == 0:
            res.losses.append((t, loss))
        if resample_every and t % resample_every == 0 and t <= resample_until * steps:
            xb, _ = world.sample(2048, rng)
            n = resample_dead(sae, xb, ~fired, rng, state)
            res.resampled += n
            fired[:] = False
    return res
