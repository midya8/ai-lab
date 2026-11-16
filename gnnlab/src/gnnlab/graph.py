"""Synthetic attributed graphs (stochastic block model) and the operators GNN layers use.

Everything is dense float64: the graphs here have a few hundred nodes, which keeps the
maths readable and the gradient checks exact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    adj: np.ndarray  # (n, n) symmetric 0/1, zero diagonal
    x: np.ndarray    # (n, f) node features
    y: np.ndarray    # (n,) integer labels
    k: int           # number of classes

    @property
    def n(self) -> int:
        return self.adj.shape[0]


@dataclass
class GraphOps:
    """Precomputed operators for one graph.

    ahat : D~^-1/2 (A+I) D~^-1/2        (GCN propagation, symmetric)
    mean : D^-1 A                       (neighbour mean, no self; zero rows for isolated nodes)
    mask : (A+I) > 0                    (who may attend to whom in GAT)
    """
    ahat: np.ndarray
    mean: np.ndarray
    mask: np.ndarray


def build_ops(adj: np.ndarray) -> GraphOps:
    adj = np.asarray(adj, dtype=np.float64)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.allclose(adj, adj.T):
        raise ValueError("adjacency must be symmetric")
    n = adj.shape[0]
    a = adj.copy()
    np.fill_diagonal(a, 0.0)
    a = (a > 0).astype(np.float64)
    at = a + np.eye(n)
    d = at.sum(1)
    dinv = 1.0 / np.sqrt(d)
    ahat = dinv[:, None] * at * dinv[None, :]
    deg = a.sum(1)
    safe = np.where(deg > 0, deg, 1.0)
    mean = a / safe[:, None]
    return GraphOps(ahat=ahat, mean=mean, mask=at > 0)


def edge_homophily(adj: np.ndarray, y: np.ndarray) -> float:
    """Fraction of edges whose endpoints share a label (nan if there are no edges)."""
    iu = np.triu_indices(adj.shape[0], 1)
    e = adj[iu] > 0
    if not e.any():
        return float("nan")
    same = (y[:, None] == y[None, :])[iu]
    return float(same[e].mean())


def sbm_graph(
    n: int = 400,
    k: int = 4,
    avg_degree: float = 10.0,
    homophily: float = 0.8,
    feat_dim: int = 16,
    feat_sep: float = 0.4,
    feat_noise: float = 1.0,
    seed: int = 0,
) -> Graph:
    """Attributed stochastic block model with a *target* edge homophily.

    Expected degree is ``avg_degree`` for every node; a fraction ``homophily`` of the
    expected edges are within-class. Features are class mean + isotropic Gaussian noise,
    so features alone carry a fixed amount of signal whatever the graph does.
    """
    if not 0.0 <= homophily <= 1.0:
        raise ValueError("homophily must be in [0, 1]")
    if k < 2 or n < 2 * k:
        raise ValueError("need k >= 2 and n >= 2k")
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.arange(n) % k)
    same = y[:, None] == y[None, :]
    np.fill_diagonal(same, False)
    diff = y[:, None] != y[None, :]
    n_same = same.sum(1).mean()   # mean same-class partners per node
    n_diff = diff.sum(1).mean()
    p_in = min(1.0, homophily * avg_degree / n_same)
    p_out = min(1.0, (1.0 - homophily) * avg_degree / n_diff)
    p = np.where(same, p_in, p_out)
    draw = rng.random((n, n)) < p
    upper = np.triu(draw, 1)
    adj = (upper | upper.T).astype(np.float64)
    means = rng.normal(size=(k, feat_dim)) * feat_sep
    x = means[y] + rng.normal(size=(n, feat_dim)) * feat_noise
    return Graph(adj=adj, x=x, y=y, k=k)


def make_split(y: np.ndarray, per_class: int = 20, n_val: int = 100, seed: int = 0):
    """Stratified split: ``per_class`` train nodes per class, ``n_val`` validation, rest test."""
    rng = np.random.default_rng(seed)
    n = len(y)
    train = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        if len(idx) < per_class + 1:
            raise ValueError("class too small for per_class")
        train.extend(rng.choice(idx, size=per_class, replace=False))
    train = np.array(sorted(train))
    rest = np.setdiff1d(np.arange(n), train)
    rest = rng.permutation(rest)
    if n_val >= len(rest):
        raise ValueError("n_val leaves no test nodes")
    val = np.sort(rest[:n_val])
    test = np.sort(rest[n_val:])
    return train, val, test
