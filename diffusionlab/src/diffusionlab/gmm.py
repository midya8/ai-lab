"""An isotropic Gaussian mixture whose noised marginals are known in closed form.

Under the DDPM forward process a component N(mu_k, s^2 I) becomes
N(sqrt(abar) mu_k, (abar s^2 + 1 - abar) I) at noise level abar, so the
marginal density, its score, and therefore the *optimal* noise predictor
eps*(x_t, t) = -sqrt(1 - abar) * score_t(x_t) are exact.  That makes it possible
to measure how far a trained network is from the best possible one, and to run
a sampler with a perfect model to separate discretisation error from
learning error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GMM:
    means: np.ndarray            # (K, D)
    std: float                   # shared isotropic std of every component
    weights: np.ndarray          # (K,)

    @property
    def K(self) -> int:
        return len(self.means)

    @property
    def D(self) -> int:
        return self.means.shape[1]

    def sample(self, n: int, rng: np.random.Generator, return_labels: bool = False):
        labels = rng.choice(self.K, size=n, p=self.weights)
        x = self.means[labels] + self.std * rng.standard_normal((n, self.D))
        return (x, labels) if return_labels else x

    # --- exact noised marginals -------------------------------------------------
    def _moments(self, abar: float):
        m = np.sqrt(abar) * self.means
        v = abar * self.std ** 2 + (1.0 - abar)
        return m, v

    def responsibilities(self, x: np.ndarray, abar: float) -> np.ndarray:
        """p(component k | x_t) at noise level abar, shape (N, K)."""
        m, v = self._moments(abar)
        d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)
        logp = np.log(self.weights)[None] - 0.5 * d2 / v
        logp -= logp.max(1, keepdims=True)
        r = np.exp(logp)
        return r / r.sum(1, keepdims=True)

    def log_density(self, x: np.ndarray, abar: float) -> np.ndarray:
        m, v = self._moments(abar)
        d2 = ((x[:, None, :] - m[None]) ** 2).sum(-1)
        logp = np.log(self.weights)[None] - 0.5 * d2 / v - 0.5 * self.D * np.log(2 * np.pi * v)
        mx = logp.max(1, keepdims=True)
        return (mx + np.log(np.exp(logp - mx).sum(1, keepdims=True)))[:, 0]

    def score(self, x: np.ndarray, abar: float) -> np.ndarray:
        """grad_x log p_t(x) of the noised mixture."""
        m, v = self._moments(abar)
        r = self.responsibilities(x, abar)
        mean_k = r @ m                      # posterior-weighted component mean
        return -(x - mean_k) / v

    def eps_star(self, x: np.ndarray, abar: float) -> np.ndarray:
        """Bayes-optimal noise prediction E[eps | x_t]."""
        return -np.sqrt(1.0 - abar) * self.score(x, abar)

    def class_score(self, x: np.ndarray, abar: float, k: int) -> np.ndarray:
        """Score of the noised single component k (a Gaussian)."""
        m, v = self._moments(abar)
        return -(x - m[k][None]) / v

    def eps_star_class(self, x: np.ndarray, abar: float, labels: np.ndarray) -> np.ndarray:
        m, v = self._moments(abar)
        return np.sqrt(1.0 - abar) * (x - m[labels]) / v

    # --- evaluation helpers -----------------------------------------------------
    def assign(self, x: np.ndarray) -> np.ndarray:
        d2 = ((x[:, None, :] - self.means[None]) ** 2).sum(-1)
        return d2.argmin(1)

    def mode_stats(self, x: np.ndarray, radius_sigmas: float = 3.0) -> dict:
        """Fraction of samples within radius_sigmas*std of a mean ("on-mode"),
        and total-variation distance between the nearest-mode histogram and the
        mixture weights."""
        d2 = ((x[:, None, :] - self.means[None]) ** 2).sum(-1)
        nearest = d2.argmin(1)
        on_mode = float((np.sqrt(d2.min(1)) <= radius_sigmas * self.std).mean())
        hist = np.bincount(nearest, minlength=self.K) / len(x)
        tv = 0.5 * float(np.abs(hist - self.weights).sum())
        return {"on_mode": on_mode, "mode_tv": tv}


def ring(K: int = 8, radius: float = 1.5, std: float = 0.3) -> GMM:
    """K equally weighted Gaussians on a circle."""
    ang = 2 * np.pi * np.arange(K) / K
    means = radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    return GMM(means, std, np.full(K, 1.0 / K))
