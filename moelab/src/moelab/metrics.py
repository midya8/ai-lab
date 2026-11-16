"""Routing diagnostics."""

from __future__ import annotations

import numpy as np


def load_stats(load: np.ndarray) -> dict:
    """Balance summaries of a load vector (shares summing to 1)."""
    load = np.asarray(load, dtype=float)
    E = load.size
    mean = 1.0 / E
    nz = load[load > 0]
    ent = float(-(nz * np.log(nz)).sum() / np.log(E)) if E > 1 else 1.0
    return {
        "cv": float(load.std() / mean),
        "max_over_mean": float(load.max() / mean),
        "dead_experts": int((load < 0.1 * mean).sum()),
        "norm_entropy": ent,
    }


def _contingency(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    t = np.zeros((ua.size, ub.size))
    np.add.at(t, (ia, ib), 1)
    return t


def routing_purity(top1: np.ndarray, cluster: np.ndarray) -> float:
    """Fraction of tokens whose cluster's modal expert is their expert (1 = clusters never split)."""
    t = _contingency(cluster, top1)
    return float(t.max(axis=1).sum() / t.sum())


def routing_nmi(top1: np.ndarray, cluster: np.ndarray) -> float:
    """Normalised mutual information between expert id and latent cluster, in [0, 1]."""
    t = _contingency(cluster, top1)
    pj = t / t.sum()
    pa, pb = pj.sum(1), pj.sum(0)
    nz = pj > 0
    mi = float((pj[nz] * np.log(pj[nz] / np.outer(pa, pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    if ha == 0 or hb == 0:
        return 0.0
    return mi / np.sqrt(ha * hb)
