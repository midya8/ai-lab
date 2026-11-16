"""Full-batch Adam training with a load-balance history."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import load_stats, routing_nmi, routing_purity


@dataclass
class Result:
    train_mse: float
    test_mse: float
    history: list = field(default_factory=list)  # (step, task_mse, load_cv, drop_rate)
    load: np.ndarray | None = None
    drop_rate: float = 0.0
    unserved: float = 0.0
    nmi: float = 0.0
    purity: float = 0.0
    dead_experts: int = 0
    load_cv: float = 0.0


def fit(net, x, y, x_test, y_test, z_train=None, steps: int = 300, lr: float = 3e-3,
        batch: int | None = None, seed: int = 0, record_every: int = 10) -> Result:
    rng = np.random.default_rng(seed)
    params = net.params()
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = []
    n = x.shape[0]
    for t in range(1, steps + 1):
        if batch is None or batch >= n:
            xb, yb = x, y
        else:
            ids = rng.choice(n, size=batch, replace=False)
            xb, yb = x[ids], y[ids]
        _, task, grads = net.loss_and_grads(xb, yb)
        for k, p in params.items():
            g = grads[k]
            m[k] = b1 * m[k] + (1 - b1) * g
            v[k] = b2 * v[k] + (1 - b2) * g * g
            mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
            p -= lr * mh / (np.sqrt(vh) + eps)       # in place: params() returns views
        if hasattr(net.layer, "update_bias"):
            net.layer.update_bias()
        if t % record_every == 0 or t == 1:
            st = getattr(net.layer, "last_stats", {})
            if st:
                hist.append((t, task, load_stats(st["load"])["cv"], st["drop_rate"]))
            else:
                hist.append((t, task, 0.0, 0.0))
    res = Result(float(np.mean((net.predict(x) - y) ** 2)),
                 float(np.mean((net.predict(x_test) - y_test) ** 2)), hist)
    st = getattr(net.layer, "last_stats", {})
    if st:
        # route the *test* set in eval mode to describe the trained router
        net.predict(x_test)
        st = net.layer.last_stats
        ls = load_stats(st["load"])
        res.load, res.drop_rate, res.unserved = st["load"], st["drop_rate"], st["token_unserved"]
        res.dead_experts, res.load_cv = ls["dead_experts"], ls["cv"]
    return res


def route_quality(net, x, z) -> tuple[float, float]:
    """(NMI, purity) of top-1 expert vs latent cluster on ``x``."""
    net.predict(x)
    top1 = net.layer.last_stats["top1"]
    return routing_nmi(top1, z), routing_purity(top1, z)
