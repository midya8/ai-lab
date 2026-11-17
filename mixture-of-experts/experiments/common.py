import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from moe import DenseFFN, MoELayer, Net, fit, make_task, sample_task
from moe.train import route_quality

D, F = 32, 32           # model width and per-expert hidden width
STEPS, LR = 400, 3e-3


def zipf(k: int, s: float = 1.0) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1) ** s
    return w / w.sum()


def trial(seed: int, layer_fn, n_clusters: int = 8, skew: bool = False, n: int = 1024):
    """Train one net on a fresh task; returns metrics dict. ``layer_fn(seed)`` builds the layer."""
    task = make_task(n_clusters=n_clusters, seed=seed)
    w = zipf(n_clusters) if skew else None
    x, y, z = sample_task(task, n, seed * 7 + 1, weights=w)
    xt, yt, zt = sample_task(task, n, seed * 7 + 2, weights=w)
    layer = layer_fn(seed)
    net = Net(8, 4, layer, D, seed=seed)
    r = fit(net, x, y, xt, yt, steps=STEPS, lr=LR, seed=seed)
    out = {"test": r.test_mse, "train": r.train_mse, "cv": r.load_cv, "dead": r.dead_experts,
           "drop": r.drop_rate, "unserved": r.unserved, "params": net.n_params()}
    if isinstance(layer, MoELayer):
        out["nmi"], out["purity"] = route_quality(net, xt, zt)
        out["active"] = layer.active_params_per_token()
    return out


def agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return v.mean(), v.std()


def fmt(rows, key, p=4):
    m, s = agg(rows, key)
    return f"{m:.{p}f}±{s:.{p}f}"
