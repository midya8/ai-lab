"""Compare a trained noise predictor with the exact Bayes-optimal one."""
from __future__ import annotations

import numpy as np

from .diffusion import q_sample
from .gmm import GMM
from .net import EpsNet
from .schedule import Schedule


def eps_error_at(net: EpsNet, gmm: GMM, sched: Schedule, t: int, n: int = 4000, seed: int = 0) -> dict:
    """At noise level t, Monte-Carlo estimates (per-coordinate means) of

      excess      E|eps_theta - eps*|^2   distance of the net from the optimum
      irreducible E|eps* - eps|^2         what even a perfect model pays
      loss        E|eps_theta - eps|^2    what training measures

    and loss ~= excess + irreducible (the cross term has zero mean, since
    E[eps | x_t] = eps*).  Unconditional prediction (null label)."""
    rng = np.random.default_rng(seed + 7919 * t)
    x0 = gmm.sample(n, rng)
    eps = rng.standard_normal(x0.shape)
    tt = np.full(n, t)
    xt = q_sample(x0, tt, eps, sched)
    pred = net.predict(xt, tt, None)
    star = gmm.eps_star(xt, sched.abar[t])
    return {
        "t": t,
        "excess": float(((pred - star) ** 2).mean()),
        "irreducible": float(((star - eps) ** 2).mean()),
        "loss": float(((pred - eps) ** 2).mean()),
    }


def eps_error_curve(net, gmm, sched, ts=None, n=4000, seed=0):
    ts = ts if ts is not None else np.unique(np.linspace(1, sched.T, 10).astype(int))
    return [eps_error_at(net, gmm, sched, int(t), n, seed) for t in ts]


def ascii_scatter(x: np.ndarray, lim: float = 2.8, width: int = 56, height: int = 24) -> str:
    """Tiny density plot for terminals / READMEs (2-D points)."""
    H, _, _ = np.histogram2d(x[:, 1], x[:, 0], bins=[height, width], range=[[-lim, lim], [-lim, lim]])
    H = H[::-1]
    shades = " .:-=+*#%@"
    if H.max() == 0:
        return "\n".join(" " * width for _ in range(height))
    idx = np.minimum((np.sqrt(H / H.max()) * (len(shades) - 1)).round().astype(int), len(shades) - 1)
    return "\n".join("".join(shades[i] for i in row) for row in idx)
