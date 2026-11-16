"""Synthetic superposition data with known ground-truth features.

``n_features`` unit-norm directions live in ``d`` dimensions (``n_features > d`` means the
features must share space, i.e. superposition). Every sample is a sparse non-negative
combination ``x = sum_i a_i f_i`` where feature ``i`` fires with probability ``p_i`` and, when
firing, has magnitude ``Uniform(mag_lo, mag_hi)``.  Because the truth is known, we can score a
sparse autoencoder on whether it *recovered the features*, not just on reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeatureWorld:
    directions: np.ndarray  # (n_features, d), unit rows
    probs: np.ndarray       # (n_features,) firing probability
    mag_lo: float = 0.5
    mag_hi: float = 1.5
    parent: np.ndarray | None = None   # (n_features,) parent index, -1 for a root; depth 1 only
    cond: np.ndarray | None = None     # (n_features,) P(fire | parent fired); roots: = probs

    @property
    def n_features(self) -> int:
        return self.directions.shape[0]

    @property
    def d(self) -> int:
        return self.directions.shape[1]

    @property
    def expected_l0(self) -> float:
        return float(self.probs.sum())

    def coherence(self) -> float:
        """Largest |cosine| between two distinct true features."""
        g = np.abs(self.directions @ self.directions.T)
        np.fill_diagonal(g, 0.0)
        return float(g.max())

    def sample(self, n: int, rng: np.random.Generator):
        """Return ``(x, a)``: activations (n, d) and the true coefficients (n, n_features)."""
        if self.parent is None:
            fire = rng.random((n, self.n_features)) < self.probs[None, :]
        else:
            fire = rng.random((n, self.n_features)) < self.cond[None, :]
            kids = np.where(self.parent >= 0)[0]
            fire[:, kids] &= fire[:, self.parent[kids]]    # a child can only fire with its parent
        mag = rng.uniform(self.mag_lo, self.mag_hi, size=fire.shape)
        a = np.where(fire, mag, 0.0)
        return a @ self.directions, a


def make_world(n_features: int, d: int, l0: float = 3.0, zipf: float = 0.0, seed: int = 0,
               mag_lo: float = 0.5, mag_hi: float = 1.5, orthogonal: bool = False) -> FeatureWorld:
    """Build a world with expected ``l0`` active features per sample.

    ``zipf`` skews firing probabilities as rank**-zipf (0 = uniform) while keeping the expected
    L0 fixed. ``orthogonal`` uses an orthonormal frame (requires ``n_features <= d``); otherwise
    directions are random Gaussian unit vectors, whose coherence grows as ``n_features / d`` does.
    """
    rng = np.random.default_rng(seed)
    if orthogonal:
        if n_features > d:
            raise ValueError("orthogonal worlds need n_features <= d")
        q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        dirs = q[:, :n_features].T.copy()
    else:
        dirs = rng.standard_normal((n_features, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    w = (np.arange(1, n_features + 1, dtype=float)) ** (-zipf)
    probs = np.minimum(w * (l0 / w.sum()), 1.0)
    # clipping at 1 loses a little mass for very skewed worlds; rescale the rest to compensate
    for _ in range(20):
        deficit = l0 - probs.sum()
        free = probs < 1.0
        if abs(deficit) < 1e-9 or not free.any():
            break
        probs[free] = np.minimum(probs[free] * (1 + deficit / probs[free].sum()), 1.0)
    return FeatureWorld(dirs, probs, mag_lo, mag_hi)


def make_hierarchical_world(n_parents: int, children_per_parent: int, d: int, p_parent: float = 0.25,
                            p_child: float = 0.5, child_cos: float = 0.0, seed: int = 0,
                            mag_lo: float = 0.5, mag_hi: float = 1.5) -> FeatureWorld:
    """Features with a hierarchy: every child implies its parent ("is a snake" implies "is an animal").

    Parents are features ``0..n_parents-1`` and fire with probability ``p_parent``. Child
    ``n_parents + i`` belongs to parent ``i // children_per_parent`` and fires with probability
    ``p_child`` *given* that its parent fired, so a child never fires alone. ``child_cos`` is the
    cosine between a child's direction and its parent's (0 = unrelated random directions).
    Marginal firing probabilities are stored in ``probs``; ``expected_l0`` counts parents and children.
    """
    if not 0.0 <= child_cos < 1.0:
        raise ValueError("child_cos must be in [0, 1)")
    rng = np.random.default_rng(seed)
    n_kids = n_parents * children_per_parent
    dirs = rng.standard_normal((n_parents + n_kids, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    parent = np.full(n_parents + n_kids, -1, dtype=int)
    for i in range(n_kids):
        pi = i // children_per_parent
        parent[n_parents + i] = pi
        if child_cos > 0:
            u = dirs[n_parents + i] - (dirs[n_parents + i] @ dirs[pi]) * dirs[pi]
            u /= np.linalg.norm(u)
            dirs[n_parents + i] = child_cos * dirs[pi] + np.sqrt(1 - child_cos ** 2) * u
    cond = np.full(n_parents + n_kids, p_parent)
    cond[n_parents:] = p_child
    probs = cond.copy()
    probs[n_parents:] = p_parent * p_child
    return FeatureWorld(dirs, probs, mag_lo, mag_hi, parent=parent, cond=cond)
