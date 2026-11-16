"""Invertible layers with hand-written backward passes.

Every layer implements
    forward(x)  -> (y, logdet)           logdet is per-example, shape (n,)
    inverse(y)  -> x
    backward(x, gy, glogdet) -> gx       accumulates parameter gradients in ``self.grads``
where ``gy`` is dL/dy and ``glogdet`` is dL/dlogdet (per-example, shape (n,)), and ``x`` is the
layer input saved from the forward pass.
"""
from __future__ import annotations

import numpy as np


class Permutation:
    """Fixed coordinate permutation (volume preserving, no parameters)."""

    def __init__(self, perm: np.ndarray):
        self.perm = np.asarray(perm)
        self.inv = np.argsort(self.perm)
        self.params: dict = {}
        self.grads: dict = {}

    def forward(self, x):
        return x[:, self.perm], np.zeros(x.shape[0])

    def inverse(self, y):
        return y[:, self.inv]

    def backward(self, x, gy, glogdet):
        return gy[:, self.inv]


class ActNorm:
    """y = x * exp(log_s) + b per coordinate; data-dependent init makes outputs zero-mean, unit-variance."""

    def __init__(self, d: int):
        self.params = {"log_s": np.zeros(d), "b": np.zeros(d)}
        self.grads = {k: np.zeros(d) for k in self.params}
        self.initialised = False

    def data_init(self, x):
        std = x.std(0) + 1e-6
        self.params["log_s"] = -np.log(std)
        self.params["b"] = -x.mean(0) / std
        self.initialised = True

    def forward(self, x):
        y = x * np.exp(self.params["log_s"]) + self.params["b"]
        return y, np.full(x.shape[0], self.params["log_s"].sum())

    def inverse(self, y):
        return (y - self.params["b"]) * np.exp(-self.params["log_s"])

    def backward(self, x, gy, glogdet):
        s = np.exp(self.params["log_s"])
        self.grads["b"] += gy.sum(0)
        self.grads["log_s"] += (gy * x * s).sum(0) + glogdet.sum()
        return gy * s


class Coupling:
    """Affine (RealNVP) or additive (NICE) coupling.

    The first ``d_a`` coordinates pass through unchanged and parameterise an elementwise
    affine map of the rest:  y_b = x_b * exp(s) + t,  [s, t] = MLP(x_a),
    s = scale * tanh(raw)  (bounded so training cannot blow up).
    ``additive=True`` fixes s = 0, which makes the layer volume preserving (logdet = 0).
    """

    def __init__(self, d: int, hidden: int, *, additive: bool = False, scale: float = 2.0,
                 seed: int = 0, zero_init: bool = True):
        if d < 2:
            raise ValueError("coupling needs d >= 2")
        self.d, self.da = d, d // 2
        self.db = d - self.da
        self.additive, self.scale = additive, scale
        rng = np.random.default_rng(seed)
        out = self.db if additive else 2 * self.db
        w2 = np.zeros((hidden, out)) if zero_init else rng.standard_normal((hidden, out)) * 0.1
        self.params = {
            "W1": rng.standard_normal((self.da, hidden)) / np.sqrt(self.da),
            "b1": np.zeros(hidden),
            "W2": w2,
            "b2": np.zeros(out),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def _net(self, xa):
        p = self.params
        h = np.tanh(xa @ p["W1"] + p["b1"])
        o = h @ p["W2"] + p["b2"]
        if self.additive:
            return h, None, o, None
        raw, t = o[:, : self.db], o[:, self.db:]
        return h, np.tanh(raw), t, raw

    def forward(self, x):
        xa, xb = x[:, : self.da], x[:, self.da:]
        h, th, t, _ = self._net(xa)
        if self.additive:
            return np.concatenate([xa, xb + t], 1), np.zeros(x.shape[0])
        s = self.scale * th
        return np.concatenate([xa, xb * np.exp(s) + t], 1), s.sum(1)

    def inverse(self, y):
        ya, yb = y[:, : self.da], y[:, self.da:]
        h, th, t, _ = self._net(ya)
        if self.additive:
            return np.concatenate([ya, yb - t], 1)
        s = self.scale * th
        return np.concatenate([ya, (yb - t) * np.exp(-s)], 1)

    def backward(self, x, gy, glogdet):
        p, g = self.params, self.grads
        xa, xb = x[:, : self.da], x[:, self.da:]
        gya, gyb = gy[:, : self.da], gy[:, self.da:]
        h, th, t, _ = self._net(xa)
        if self.additive:
            go = gyb
            gxb = gyb
        else:
            s = self.scale * th
            e = np.exp(s)
            gxb = gyb * e
            gs = gyb * xb * e + glogdet[:, None]
            graw = gs * self.scale * (1 - th ** 2)
            go = np.concatenate([graw, gyb], 1)
        g["W2"] += h.T @ go
        g["b2"] += go.sum(0)
        gh = go @ p["W2"].T
        gpre = gh * (1 - h ** 2)
        g["W1"] += xa.T @ gpre
        g["b1"] += gpre.sum(0)
        gxa = gya + gpre @ p["W1"].T
        return np.concatenate([gxa, gxb], 1)
