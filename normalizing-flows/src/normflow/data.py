"""2-D toy densities.  Some have an exact log-density (``oracle``), so a trained flow's
NLL can be compared with the true entropy rather than with another model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

LOG2PI = float(np.log(2 * np.pi))


@dataclass
class Dataset:
    name: str
    sample: Callable[[int, int], np.ndarray]                    # (n, seed) -> (n, 2)
    log_prob: Callable[[np.ndarray], np.ndarray] | None = None  # exact density if known
    centers: np.ndarray | None = None                           # mode centres, for mass-between-modes checks
    sigma: float | None = None


def _logsumexp(a, axis):
    m = a.max(axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def gaussian_ring(k: int = 8, radius: float = 2.0, sigma: float = 0.2) -> Dataset:
    """k equal-weight isotropic Gaussians on a circle.  Exact density; k separated modes
    make a connected unimodal base distribution genuinely struggle."""
    ang = 2 * np.pi * np.arange(k) / k
    centers = radius * np.stack([np.cos(ang), np.sin(ang)], 1)

    def sample(n, seed=0):
        rng = np.random.default_rng(seed)
        c = rng.integers(k, size=n)
        return centers[c] + sigma * rng.standard_normal((n, 2))

    def log_prob(x):
        d2 = ((x[:, None, :] - centers[None]) ** 2).sum(2)
        return _logsumexp(-0.5 * d2 / sigma ** 2, 1) - np.log(k) - LOG2PI - 2 * np.log(sigma)

    return Dataset(f"ring{k}", sample, log_prob, centers, sigma)


def two_moons(noise: float = 0.1) -> Dataset:
    def sample(n, seed=0):
        rng = np.random.default_rng(seed)
        t = rng.uniform(0, np.pi, n)
        up = rng.random(n) < 0.5
        x = np.where(up, np.cos(t), 1 - np.cos(t))
        y = np.where(up, np.sin(t), 0.5 - np.sin(t))
        return np.stack([x, y], 1) + noise * rng.standard_normal((n, 2))

    return Dataset("moons", sample)


def checkerboard(cells: int = 4) -> Dataset:
    """Uniform on the black squares of a cells x cells board over [-2, 2]^2 (density has hard edges)."""
    def sample(n, seed=0):
        rng = np.random.default_rng(seed)
        out = []
        while sum(len(o) for o in out) < n:
            p = rng.uniform(-2, 2, size=(4 * n, 2))
            ij = np.floor((p + 2) / (4 / cells)).astype(int)
            out.append(p[(ij.sum(1) % 2) == 0])
        return np.concatenate(out)[:n]

    def log_prob(x):
        inside = (np.abs(x) < 2).all(1)
        ij = np.floor((x + 2) / (4 / cells)).astype(int)
        black = (ij.sum(1) % 2) == 0
        area = (4.0 / cells) ** 2 * (cells * cells / 2)
        return np.where(inside & black, -np.log(area), -np.inf)

    return Dataset(f"checker{cells}", sample, log_prob)


def spiral(noise: float = 0.05, turns: float = 1.5) -> Dataset:
    def sample(n, seed=0):
        rng = np.random.default_rng(seed)
        t = np.sqrt(rng.uniform(0.02, 1, n)) * turns * 2 * np.pi
        x = np.stack([t * np.cos(t), t * np.sin(t)], 1) / (turns * 2 * np.pi) * 3
        return x + noise * rng.standard_normal((n, 2))

    return Dataset("spiral", sample)


DATASETS = {"ring8": gaussian_ring, "moons": two_moons, "checker4": checkerboard, "spiral": spiral}
