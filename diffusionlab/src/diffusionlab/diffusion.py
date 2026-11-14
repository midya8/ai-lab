"""DDPM training and DDIM / ancestral sampling (with classifier-free guidance)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .net import Adam, EpsNet
from .schedule import Schedule

EpsFn = Callable[[np.ndarray, int, Optional[np.ndarray]], np.ndarray]


def q_sample(x0: np.ndarray, t: np.ndarray, eps: np.ndarray, sched: Schedule) -> np.ndarray:
    ab = sched.abar[t][:, None]
    return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * eps


def train(net: EpsNet, sampler_fn, sched: Schedule, steps: int = 4000, batch: int = 256,
          lr: float = 2e-3, p_uncond: float = 0.1, ema_decay: float = 0.995,
          seed: int = 0, log_every: int = 0) -> list[float]:
    """Train on the epsilon-prediction loss with t ~ U{1..T}.

    sampler_fn(n, rng) -> (x0, labels).  Data are drawn fresh every step (an
    effectively infinite training set), so there is no memorisation to
    worry about.  With p_uncond > 0 labels are replaced by null (-1) so one
    network serves both conditional and unconditional sampling.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(net.params, lr=lr)
    hist = []
    for it in range(steps):
        x0, labels = sampler_fn(batch, rng)
        t = rng.integers(1, sched.T + 1, size=batch)
        eps = rng.standard_normal(x0.shape)
        xt = q_sample(x0, t, eps, sched)
        lab = labels.copy() if labels is not None else None
        if lab is not None and p_uncond > 0:
            lab[rng.random(batch) < p_uncond] = -1
        loss, grads = net.loss_and_grads(xt, t, lab, eps)
        cosine = 0.5 * (1 + np.cos(np.pi * it / steps))
        opt.step(grads, lr=lr * (0.05 + 0.95 * cosine))
        net.update_ema(min(ema_decay, (1 + it) / (10 + it)))
        hist.append(loss)
        if log_every and (it + 1) % log_every == 0:
            print(f"step {it+1:5d} loss {np.mean(hist[-log_every:]):.4f}")
    return hist


def timestep_grid(T: int, steps: int) -> np.ndarray:
    """Descending grid T = tau_0 > ... > tau_steps = 0 with `steps` transitions."""
    g = np.unique(np.round(np.linspace(T, 0, steps + 1)).astype(int))[::-1]
    return g


def sample(eps_fn: EpsFn, sched: Schedule, n: int, D: int, steps: int, eta: float = 0.0,
           rng: Optional[np.random.Generator] = None, labels: Optional[np.ndarray] = None,
           x_T: Optional[np.ndarray] = None) -> np.ndarray:
    """Generalised DDIM (Song et al.).  eta=0 is deterministic DDIM, eta=1 with
    steps=T is the DDPM ancestral sampler (same marginals and posterior variance)."""
    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal((n, D)) if x_T is None else x_T.copy()
    grid = timestep_grid(sched.T, steps)
    for t, s in zip(grid[:-1], grid[1:]):
        ab_t, ab_s = sched.abar[t], sched.abar[s]
        eps = eps_fn(x, int(t), labels)
        x0_hat = (x - np.sqrt(1.0 - ab_t) * eps) / np.sqrt(ab_t)
        sigma = eta * np.sqrt((1.0 - ab_s) / (1.0 - ab_t)) * np.sqrt(1.0 - ab_t / ab_s)
        dir_xt = np.sqrt(np.maximum(1.0 - ab_s - sigma ** 2, 0.0)) * eps
        x = np.sqrt(ab_s) * x0_hat + dir_xt
        if sigma > 0:
            x = x + sigma * rng.standard_normal(x.shape)
    return x


def net_eps_fn(net: EpsNet, guidance: float = 1.0, use_ema: bool = True) -> EpsFn:
    """eps(x, t, labels) from a network.  With labels and guidance != 1 this is
    classifier-free guidance: eps_u + guidance * (eps_c - eps_u).  Labels None
    (or guidance == 0) gives the unconditional prediction."""
    def fn(x, t, labels):
        tt = np.full(len(x), t)
        if labels is None:
            return net.predict(x, tt, None, use_ema)
        c = net.predict(x, tt, labels, use_ema)
        if guidance == 1.0:
            return c
        u = net.predict(x, tt, None, use_ema)
        return u + guidance * (c - u)
    return fn


def oracle_eps_fn(gmm, sched: Schedule) -> EpsFn:
    """Exact Bayes-optimal noise predictor (unconditional, or class-conditional if labels given)."""
    def fn(x, t, labels):
        ab = sched.abar[t]
        if labels is None:
            return gmm.eps_star(x, ab)
        return gmm.eps_star_class(x, ab, np.asarray(labels))
    return fn
