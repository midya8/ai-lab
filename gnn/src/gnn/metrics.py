from __future__ import annotations

import numpy as np

from .graph import GraphOps


def accuracy(logits: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    return float((logits[idx].argmax(1) == y[idx]).mean())


def dirichlet_energy(h: np.ndarray, ops: GraphOps) -> float:
    """tr(H^T (I - Ahat) H) / ||H||_F^2, in [0, 2].

    Equals half the sum over (A+I) edges of ||h_i/sqrt(d_i) - h_j/sqrt(d_j)||^2, normalised by
    scale. It is 0 exactly when every feature column is proportional to sqrt(d~): the
    "oversmoothed" subspace GCN propagation converges to.
    """
    fro = float((h * h).sum())
    if fro == 0.0:
        return 0.0
    return float(((h * h).sum() - np.einsum("if,ij,jf->", h, ops.ahat, h)) / fro)


def mean_pairwise_cosine(h: np.ndarray) -> float:
    """Average cosine similarity over distinct node pairs (1.0 = all nodes point the same way)."""
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    u = h / np.where(norms > 0, norms, 1.0)
    n = h.shape[0]
    total = (u.sum(0) @ u.sum(0)) - (u * u).sum()
    return float(total / (n * (n - 1)))
