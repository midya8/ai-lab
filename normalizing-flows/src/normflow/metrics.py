from __future__ import annotations

import numpy as np

from .data import Dataset
from .flow import Flow


def nll_gap(flow: Flow, ds: Dataset, n: int = 20000, seed: int = 10_000) -> dict:
    """Held-out flow NLL, exact data entropy (Monte-Carlo, same samples) and their difference.

    The gap is a Monte-Carlo estimate of KL(data || flow) in nats; it is >= 0 in expectation
    and tells you how far the flow is from the best possible model.  Needs ds.log_prob.
    """
    if ds.log_prob is None:
        raise ValueError(f"{ds.name} has no exact density")
    x = ds.sample(n, seed)
    flow_nll = flow.nll(x)
    true_nll = float(-ds.log_prob(x).mean())
    return {"flow_nll": flow_nll, "true_nll": true_nll, "gap": flow_nll - true_nll}


def bridge_mass(samples: np.ndarray, centers: np.ndarray, sigma: float, n_sigma: float = 3.5) -> float:
    """Fraction of samples farther than n_sigma*sigma from every mode centre: probability mass the
    model places *between* modes, where the true density is ~0 (topological mismatch)."""
    d = np.sqrt(((samples[:, None, :] - centers[None]) ** 2).sum(2)).min(1)
    return float((d > n_sigma * sigma).mean())


def density_grid_integral(flow: Flow, lim: float = 6.0, res: int = 400) -> float:
    """Riemann-sum integral of exp(log_prob) over [-lim, lim]^2: should be ~1 for a normalised flow
    (up to mass outside the window)."""
    g = np.linspace(-lim, lim, res)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    p = np.exp(flow.log_prob(pts))
    return float(p.sum() * (g[1] - g[0]) ** 2)
