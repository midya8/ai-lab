"""CNN layers in pure NumPy with hand-written forward and backward passes.

Every layer exposes ``params()`` / ``grads()`` (dicts of arrays that are updated
*in place*, so an optimiser can hold references) plus ``forward`` / ``backward``.
Tensors are NCHW. All layers are dtype-agnostic: they follow the dtype of their
parameters / inputs, so the test-suite can run them in float64 for tight
finite-difference gradient checks.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Layer:
    training = True

    def params(self) -> dict:
        return {}

    def grads(self) -> dict:
        return {}

    def buffers(self) -> dict:
        """Non-trainable state that must survive save/load (e.g. running statistics)."""
        return {}

    def forward(self, x):  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, g):  # pragma: no cover - interface
        raise NotImplementedError


class Normalize(Layer):
    """Fixed affine input standardisation, so attacks/explanations work in raw pixel space."""

    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, x):
        return (x - self.mean) / self.std

    def backward(self, g):
        return g / self.std


class Conv2D(Layer):
    """Stride-1 convolution via im2col (a strided window view + one matmul)."""

    def __init__(self, in_ch, out_ch, k=3, pad=1, rng=None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.k, self.pad, self.in_ch, self.out_ch = k, pad, in_ch, out_ch
        fan_in = in_ch * k * k
        self.W = (rng.standard_normal((out_ch, in_ch, k, k)) * np.sqrt(2.0 / fan_in)).astype(dtype)
        self.b = np.zeros(out_ch, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def forward(self, x):
        n, c, h, w = x.shape
        k, p = self.k, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        win = sliding_window_view(xp, (k, k), axis=(2, 3))  # N,C,Ho,Wo,k,k
        ho, wo = win.shape[2], win.shape[3]
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)
        out = cols @ self.W.reshape(self.out_ch, -1).T + self.b
        out = out.reshape(n, ho, wo, self.out_ch).transpose(0, 3, 1, 2)
        self._cache = (cols, x.shape, ho, wo)
        return np.ascontiguousarray(out)

    def backward(self, g):
        cols, (n, c, h, w), ho, wo = self._cache
        k, p = self.k, self.pad
        gf = g.transpose(0, 2, 3, 1).reshape(-1, self.out_ch)
        self.dW[...] = (gf.T @ cols).reshape(self.W.shape)
        self.db[...] = gf.sum(axis=0)
        dcols = (gf @ self.W.reshape(self.out_ch, -1)).reshape(n, ho, wo, c, k, k)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=dcols.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i:i + ho, j:j + wo] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p:p + h, p:p + w]


class ReLU(Layer):
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, g):
        return g * self._mask


class MaxPool2D(Layer):
    """2x2 / stride-2 max pooling. The gradient goes to exactly one argmax per window."""

    def forward(self, x):
        n, c, h, w = x.shape
        h2, w2 = h // 2, w // 2
        r = x[:, :, :2 * h2, :2 * w2].reshape(n, c, h2, 2, w2, 2)
        r = r.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h2, w2, 4)
        idx = r.argmax(axis=-1)
        self._idx, self._shape = idx, x.shape
        return np.take_along_axis(r, idx[..., None], axis=-1)[..., 0]

    def backward(self, g):
        n, c, h, w = self._shape
        h2, w2 = h // 2, w // 2
        dr = np.zeros((n, c, h2, w2, 4), dtype=g.dtype)
        np.put_along_axis(dr, self._idx[..., None], g[..., None], axis=-1)
        dr = dr.reshape(n, c, h2, w2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, 2 * h2, 2 * w2)
        if (2 * h2, 2 * w2) == (h, w):
            return dr
        out = np.zeros(self._shape, dtype=g.dtype)
        out[:, :, :2 * h2, :2 * w2] = dr
        return out


class Flatten(Layer):
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, g):
        return g.reshape(self._shape)


class Dense(Layer):
    def __init__(self, n_in, n_out, rng=None, dtype=np.float32):
        rng = rng if rng is not None else np.random.default_rng(0)
        self.W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(dtype)
        self.b = np.zeros(n_out, dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def params(self):
        return {"W": self.W, "b": self.b}

    def grads(self):
        return {"W": self.dW, "b": self.db}

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW[...] = self._x.T @ g
        self.db[...] = g.sum(axis=0)
        return g @ self.W.T


class Dropout(Layer):
    """Inverted dropout; identity in eval mode."""

    def __init__(self, p=0.0, rng=None):
        self.p = float(p)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._mask = None

    def forward(self, x):
        if not self.training or self.p <= 0:
            self._mask = None
            return x
        keep = 1.0 - self.p
        self._mask = ((self.rng.random(x.shape) < keep) / keep).astype(x.dtype)
        return x * self._mask

    def backward(self, g):
        return g if self._mask is None else g * self._mask


class BatchNorm2D(Layer):
    """Per-channel batch normalisation (Ioffe & Szegedy) for NCHW tensors.

    Training mode normalises with the batch statistics and updates exponential running
    averages; eval mode uses the running averages, so it is a fixed per-channel affine map
    (its backward is then just a scale, which attacks and Grad-CAM rely on).
    """

    def __init__(self, ch, momentum=0.1, eps=1e-5, dtype=np.float32):
        self.ch, self.momentum, self.eps = ch, float(momentum), float(eps)
        self.gamma = np.ones(ch, dtype=dtype)
        self.beta = np.zeros(ch, dtype=dtype)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)
        self.running_mean = np.zeros(ch, dtype=dtype)
        self.running_var = np.ones(ch, dtype=dtype)
        self._cache = None

    def params(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def grads(self):
        return {"gamma": self.dgamma, "beta": self.dbeta}

    def buffers(self):
        return {"running_mean": self.running_mean, "running_var": self.running_var}

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))
            m = self.momentum
            n = x.size / self.ch
            self.running_mean *= 1 - m
            self.running_mean += m * mean
            self.running_var *= 1 - m
            self.running_var += m * var * (n / max(n - 1, 1))   # unbiased estimate for inference
        else:
            mean, var = self.running_mean, self.running_var
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mean[None, :, None, None]) * inv[None, :, None, None]
        self._cache = (xhat, inv, self.training)
        return xhat * self.gamma[None, :, None, None] + self.beta[None, :, None, None]

    def backward(self, g):
        xhat, inv, was_training = self._cache
        self.dgamma[...] = (g * xhat).sum(axis=(0, 2, 3))
        self.dbeta[...] = g.sum(axis=(0, 2, 3))
        gx = g * self.gamma[None, :, None, None]
        if not was_training:
            return gx * inv[None, :, None, None]
        m = g.shape[0] * g.shape[2] * g.shape[3]
        return (inv[None, :, None, None] / m) * (
            m * gx - gx.sum(axis=(0, 2, 3), keepdims=True)
            - xhat * (gx * xhat).sum(axis=(0, 2, 3), keepdims=True))


class GlobalAvgPool(Layer):
    """Mean over the spatial dimensions: (N,C,H,W) -> (N,C). Gives translation invariance."""

    def forward(self, x):
        self._shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, g):
        n, c, h, w = self._shape
        return np.broadcast_to((g / (h * w))[:, :, None, None], self._shape).copy()
