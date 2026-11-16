"""Full-batch gradient descent on MSE, optionally in the "lazy" alpha-scaled form."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .net import MLP


@dataclass
class TrainResult:
    net: MLP
    net0: MLP
    alpha: float
    center: bool
    losses: list = field(default_factory=list)
    steps: list = field(default_factory=list)

    def predict(self, X) -> np.ndarray:
        f = self.net(X)
        if self.center:
            f = f - self.net0(X)
        return self.alpha * f


def train_gd(net: MLP, X, y, lr: float, steps: int, alpha: float = 1.0,
             center: bool = False, record_every: int = 0, on_record=None) -> TrainResult:
    """Train a copy of `net` with plain GD on (1/2n) sum (alpha*(f - center*f0) - y)^2.

    The step is lr / alpha^2, so as alpha grows the *function-space* dynamics stay the
    same while the parameters move less: the lazy / NTK regime (Chizat & Bach 2019).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float)
    n = len(y)
    work = net.copy()
    res = TrainResult(work, net.copy(), alpha, center)
    f0 = net(X) if center else 0.0
    step_lr = lr / alpha ** 2
    for t in range(steps + 1):
        f, cache = work.forward(X)
        out = alpha * (f - f0)
        r = out - y
        if record_every and (t % record_every == 0 or t == steps):
            res.losses.append(float(0.5 * np.mean(r ** 2)))
            res.steps.append(t)
            if on_record is not None:
                on_record(t, work)
        if t == steps:
            break
        dW, db = work.grads(cache, alpha * r / n)
        for k in range(work.n_layers):
            work.W[k] = work.W[k] - step_lr * dW[k]
            work.b[k] = work.b[k] - step_lr * db[k]
    return res


def kernel_change(K_now: np.ndarray, K_init: np.ndarray) -> float:
    """Relative Frobenius movement of the kernel."""
    return float(np.linalg.norm(K_now - K_init) / np.linalg.norm(K_init))
