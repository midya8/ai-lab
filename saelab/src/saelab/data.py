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
        fire = rng.random((n, self.n_features)) < self.probs[None, :]
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
