from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flow import Flow


class Adam:
    def __init__(self, params: dict, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict, lr: float | None = None, clip: float | None = None):
        self.t += 1
        lr = self.lr if lr is None else lr
        if clip is not None:
            gn = np.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
            f = min(1.0, clip / (gn + 1e-12))
        else:
            f = 1.0
        for k, g in grads.items():
            g = g * f
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            params[k] -= lr * (self.m[k] / (1 - self.b1 ** self.t)) / (np.sqrt(self.v[k] / (1 - self.b2 ** self.t)) + self.eps)


@dataclass
class FitResult:
    losses: list = field(default_factory=list)   # (step, minibatch nll)
    val: list = field(default_factory=list)      # (step, validation nll)


def fit(flow: Flow, x_train: np.ndarray, *, steps: int = 2000, batch: int = 256, lr: float = 2e-3,
        seed: int = 0, x_val: np.ndarray | None = None, eval_every: int = 250, clip: float = 10.0,
        log_every: int = 50) -> FitResult:
    rng = np.random.default_rng(seed)
    flow.data_init(x_train[: min(len(x_train), 2048)])
    params = flow.named_params()
    opt = Adam(params, lr)
    res = FitResult()
    n = len(x_train)
    for t in range(1, steps + 1):
        idx = rng.integers(n, size=batch)
        loss, grads = flow.loss_and_grads(x_train[idx])
        cur = lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * (t - 1) / steps)))
        opt.step(params, grads, cur, clip)
        if t % log_every == 0:
            res.losses.append((t, loss))
        if x_val is not None and (t % eval_every == 0 or t == steps):
            res.val.append((t, flow.nll(x_val)))
    return res
