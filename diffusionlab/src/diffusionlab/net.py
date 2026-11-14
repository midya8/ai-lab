"""A small conditional noise-prediction MLP with hand-written backprop (NumPy).

Input features: x (D) | sinusoidal time features (2F) | class one-hot (C, all
zeros = "null" label used for unconditional / classifier-free training).
Hidden activation: SiLU.  Output: predicted noise, D values.
"""
from __future__ import annotations

import numpy as np


def time_features(t: np.ndarray, T: int, n_freq: int = 8) -> np.ndarray:
    u = np.asarray(t, dtype=np.float64) / T                       # (N,) in [0, 1]
    freqs = 2.0 ** np.arange(n_freq) * np.pi
    ang = u[:, None] * freqs[None]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)


def _silu(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return z * s, s * (1.0 + z * (1.0 - s))      # value, derivative


class EpsNet:
    def __init__(self, D: int = 2, hidden: int = 128, depth: int = 3, n_classes: int = 0,
                 T: int = 100, n_freq: int = 8, seed: int = 0):
        self.D, self.C, self.T, self.n_freq = D, n_classes, T, n_freq
        rng = np.random.default_rng(seed)
        sizes = [D + 2 * n_freq + n_classes] + [hidden] * depth + [D]
        self.params: list[np.ndarray] = []
        for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
            scale = np.sqrt(2.0 / a) if i < len(sizes) - 2 else 0.01   # near-zero output init
            self.params += [rng.normal(0, scale, (a, b)), np.zeros(b)]
        self.ema = [p.copy() for p in self.params]

    # --- helpers ---------------------------------------------------------------
    def featurize(self, x, t, labels=None):
        parts = [x, time_features(t, self.T, self.n_freq)]
        if self.C:
            oh = np.zeros((len(x), self.C))
            if labels is not None:
                lab = np.asarray(labels)
                ok = lab >= 0                                  # -1 means null
                oh[np.arange(len(x))[ok], lab[ok]] = 1.0
            parts.append(oh)
        return np.concatenate(parts, axis=1)

    def _forward(self, feats, params):
        acts, pres, h = [feats], [], feats
        n = len(params) // 2
        for i in range(n):
            z = h @ params[2 * i] + params[2 * i + 1]
            if i < n - 1:
                pres.append(z)
                h = _silu(z)[0]
            else:
                h = z
            acts.append(h)
        return h, acts, pres

    # --- public API -------------------------------------------------------------
    def predict(self, x, t, labels=None, use_ema: bool = True) -> np.ndarray:
        feats = self.featurize(x, t, labels)
        return self._forward(feats, self.ema if use_ema else self.params)[0]

    def loss_and_grads(self, x, t, labels, target):
        """Mean squared error against `target` (elementwise mean) and parameter grads."""
        feats = self.featurize(x, t, labels)
        out, acts, pres = self._forward(feats, self.params)
        diff = out - target
        loss = float((diff ** 2).mean())
        g = 2.0 * diff / diff.size
        grads = [None] * len(self.params)
        n = len(self.params) // 2
        for i in reversed(range(n)):
            grads[2 * i] = acts[i].T @ g
            grads[2 * i + 1] = g.sum(0)
            if i > 0:
                g = (g @ self.params[2 * i].T) * _silu(pres[i - 1])[1]
        return loss, grads

    def update_ema(self, decay: float):
        for e, p in zip(self.ema, self.params):
            e *= decay
            e += (1.0 - decay) * p

    def n_params(self) -> int:
        return int(sum(p.size for p in self.params))


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * g * g
            p -= lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)
