"""A normalizing flow: a stack of invertible layers onto a standard-normal base."""
from __future__ import annotations

import numpy as np

from .layers import ActNorm, Coupling, InvLinear, Permutation

LOG2PI = float(np.log(2 * np.pi))


class Flow:
    def __init__(self, d: int, n_layers: int = 6, hidden: int = 64, *, additive: bool = False,
                 actnorm: bool = True, seed: int = 0, zero_init: bool = True, scale: float = 2.0, mixing: str = "perm"):
        if mixing not in ("perm", "linear"):
            raise ValueError("mixing must be 'perm' or 'linear'")
        self.d = d
        rng = np.random.default_rng(seed)
        self.layers: list = []
        if actnorm:
            self.layers.append(ActNorm(d))
        for i in range(n_layers):
            self.layers.append(Coupling(d, hidden, additive=additive, scale=scale,
                                        seed=int(rng.integers(1 << 30)), zero_init=zero_init))
            # alternate which coordinates condition: reverse for d=2, random permutation otherwise
            perm = np.arange(d)[::-1] if d == 2 else rng.permutation(d)
            self.layers.append(InvLinear(d, seed=int(rng.integers(1 << 30))) if mixing == "linear"
                               else Permutation(perm))

    # ----- parameters as a flat dict for the optimiser -----
    def named_params(self) -> dict:
        out = {}
        for i, l in enumerate(self.layers):
            for k, v in l.params.items():
                out[f"{i}.{k}"] = v
        return out

    def named_grads(self) -> dict:
        out = {}
        for i, l in enumerate(self.layers):
            for k, v in l.grads.items():
                out[f"{i}.{k}"] = v
        return out

    def n_params(self) -> int:
        return int(sum(v.size for v in self.named_params().values()))

    def zero_grads(self) -> None:
        for l in self.layers:
            for k in l.grads:
                l.grads[k][...] = 0.0

    def data_init(self, x) -> None:
        for l in self.layers:
            if isinstance(l, ActNorm) and not l.initialised:
                l.data_init(x)
            x, _ = l.forward(x)

    # ----- density -----
    def forward(self, x):
        """x -> (z, logdet, saved layer inputs)."""
        saved = []
        logdet = np.zeros(x.shape[0])
        for l in self.layers:
            saved.append(x)
            x, ld = l.forward(x)
            logdet = logdet + ld
        return x, logdet, saved

    def inverse(self, z):
        for l in reversed(self.layers):
            z = l.inverse(z)
        return z

    def log_prob(self, x) -> np.ndarray:
        z, logdet, _ = self.forward(x)
        return -0.5 * (z ** 2).sum(1) - 0.5 * self.d * LOG2PI + logdet

    def nll(self, x) -> float:
        """Mean negative log-likelihood in nats per example."""
        return float(-self.log_prob(x).mean())

    def loss_and_grads(self, x):
        n = x.shape[0]
        self.zero_grads()
        z, logdet, saved = self.forward(x)
        loss = float((0.5 * (z ** 2).sum(1) + 0.5 * self.d * LOG2PI - logdet).mean())
        g = z / n
        gld = np.full(n, -1.0 / n)
        for l, xin in zip(reversed(self.layers), reversed(saved)):
            g = l.backward(xin, g, gld)
        return loss, self.named_grads()

    def sample(self, n: int, *, seed: int = 0, temperature: float = 1.0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.inverse(temperature * rng.standard_normal((n, self.d)))
