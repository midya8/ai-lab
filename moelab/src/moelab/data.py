"""A synthetic task in which specialising experts is the right thing to do.

Inputs come from ``K`` well-separated Gaussian clusters; each cluster has its *own*
target function ``y = tanh(x W_k)``.  A router that recovers the cluster id can hand
each cluster to its own expert, so the task isolates *routing* behaviour (collapse,
load balance, capacity drops) from any claim about language or images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Task:
    centers: np.ndarray  # (K, d_in)
    maps: np.ndarray     # (K, d_in, d_out)
    sigma: float
    noise: float

    @property
    def n_clusters(self) -> int:
        return self.centers.shape[0]


def make_task(n_clusters: int = 8, d_in: int = 8, d_out: int = 4, sep: float = 3.0,
              sigma: float = 0.5, noise: float = 0.02, seed: int = 0) -> Task:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_clusters, d_in)) * sep
    scale = 1.0 / np.sqrt(d_in * (sep ** 2 + sigma ** 2))
    maps = rng.normal(size=(n_clusters, d_in, d_out)) * scale
    return Task(centers, maps, sigma, noise)


def sample_task(task: Task, n: int, seed: int, weights: np.ndarray | None = None):
    """Draw ``n`` points; ``weights`` optionally skews the cluster frequencies."""
    rng = np.random.default_rng(seed)
    z = rng.choice(task.n_clusters, size=n, p=weights)
    x = task.centers[z] + rng.normal(size=(n, task.centers.shape[1])) * task.sigma
    pre = np.einsum("nd,ndo->no", x, task.maps[z])
    y = np.tanh(pre * 1.5) + rng.normal(size=pre.shape) * task.noise
    return x, y, z
