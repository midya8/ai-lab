"""Layers with hand-written forward and backward passes (NumPy, float64).

Every layer keeps its parameters in ``self.p`` and the gradients of the last
backward call in ``self.g``. ``backward`` takes the upstream gradient and returns
the gradient with respect to the layer input.
"""
from __future__ import annotations

import numpy as np

from .scan import scan, reverse_scan


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softplus(x):
    return np.logaddexp(0.0, x)


def silu(x):
    return x * sigmoid(x)


def silu_backward(dy, x):
    s = sigmoid(x)
    return dy * (s + x * s * (1.0 - s))


class Linear:
    def __init__(self, d_in, d_out, rng, scale=None):
        scale = 1.0 / np.sqrt(d_in) if scale is None else scale
        self.p = {"W": rng.normal(0.0, scale, (d_in, d_out))}
        self.g = {}

    def forward(self, x):
        self.x = x
        return x @ self.p["W"]

    def backward(self, dy):
        x2 = self.x.reshape(-1, self.x.shape[-1])
        self.g = {"W": x2.T @ dy.reshape(-1, dy.shape[-1])}
        return dy @ self.p["W"].T


class RMSNorm:
    def __init__(self, d, eps=1e-6):
        self.p = {"w": np.ones(d)}
        self.g = {}
        self.eps = eps

    def forward(self, x):
        r = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        self.x, self.r = x, r
        return x * r * self.p["w"]

    def backward(self, dy):
        x, r, w = self.x, self.r, self.p["w"]
        self.g = {"w": (dy * x * r).reshape(-1, x.shape[-1]).sum(0)}
        dxh = dy * w
        return r * dxh - x * r ** 3 * np.mean(dxh * x, axis=-1, keepdims=True)


class CausalDWConv:
    """Depthwise causal 1-D convolution: y_t = b + sum_k w[:, k] * x_{t-(K-1)+k}."""

    def __init__(self, d, k, rng):
        self.k = k
        self.p = {"w": rng.normal(0.0, 1.0 / np.sqrt(k), (d, k)), "b": np.zeros(d)}
        self.g = {}

    def forward(self, x):
        B, T, D = x.shape
        K = self.k
        xp = np.concatenate([np.zeros((B, K - 1, D)), x], axis=1)
        self.xp, self.T = xp, T
        y = np.zeros_like(x) + self.p["b"]
        for j in range(K):
            y += self.p["w"][:, j] * xp[:, j:j + T]
        return y

    def backward(self, dy):
        K, T = self.k, self.T
        dw = np.zeros_like(self.p["w"])
        dxp = np.zeros_like(self.xp)
        for j in range(K):
            dw[:, j] = (dy * self.xp[:, j:j + T]).sum((0, 1))
            dxp[:, j:j + T] += dy * self.p["w"][:, j]
        self.g = {"w": dw, "b": dy.sum((0, 1))}
        return dxp[:, K - 1:]

    def step(self, x, buf):
        """One token: x (B, D); buf (B, K-1, D) holds the previous inputs. Returns (y, new_buf)."""
        full = np.concatenate([buf, x[:, None]], axis=1)  # (B, K, D)
        y = self.p["b"] + np.einsum("dk,bkd->bd", self.p["w"], full)
        return y, full[:, 1:]


class SSM:
    """Diagonal state-space layer, ``h_t = Abar_t * h_{t-1} + Bbar_t * u_t``, ``y_t = C_t . h_t + D u_t``.

    ``selective=True``  (Mamba-style): the step size ``delta`` and the matrices ``B`` and ``C``
    are functions of the input, so the layer can choose per token what to write and what to forget.
    ``selective=False`` (S4D-style, LTI): ``delta``, ``B``, ``C`` are fixed parameters, so the layer
    is a fixed convolution of its input.

    A = -exp(A_log) is diagonal, discretised by zero-order hold for A and Euler for B
    (``Bbar = delta * B``), the same shortcut the Mamba reference code takes.
    """

    def __init__(self, d, n, rng, selective=True, dt_min=1e-3, dt_max=0.1, scan_method="sequential"):
        self.d, self.n, self.selective, self.scan_method = d, n, selective, scan_method
        dt = np.exp(rng.uniform(np.log(dt_min), np.log(dt_max), d))
        p = {
            "A_log": np.log(np.tile(np.arange(1, n + 1, dtype=float), (d, 1))),
            "Dskip": np.ones(d),
            "b_dt": dt + np.log(-np.expm1(-dt)),  # inverse softplus
        }
        if selective:
            p["W_dt"] = rng.normal(0.0, 0.5 / np.sqrt(d), (d, d))
            p["W_B"] = rng.normal(0.0, 1.0 / np.sqrt(d), (d, n))
            p["W_C"] = rng.normal(0.0, 1.0 / np.sqrt(d), (d, n))
        else:
            p["B"] = rng.normal(0.0, 1.0, (d, n))
            p["C"] = rng.normal(0.0, 1.0 / np.sqrt(n), (d, n))
        self.p, self.g = p, {}

    # -- shared pieces -------------------------------------------------
    def _coeffs(self, u):
        p = self.p
        if self.selective:
            z = u @ p["W_dt"] + p["b_dt"]
            Bt = u @ p["W_B"]  # (B, T, N)
            Ct = u @ p["W_C"]
            Bf, Cf = Bt[..., None, :], Ct[..., None, :]
        else:
            z = np.broadcast_to(p["b_dt"], u.shape)
            Bf = p["B"][None, None] if u.ndim == 3 else p["B"][None]
            Cf = p["C"][None, None] if u.ndim == 3 else p["C"][None]
        delta = softplus(z)
        A = -np.exp(p["A_log"])
        Abar = np.exp(delta[..., None] * A)
        return z, delta, A, Abar, Bf, Cf

    def forward(self, u, h0=None):
        z, delta, A, Abar, Bf, Cf = self._coeffs(u)
        Bu = delta[..., None] * Bf * u[..., None]
        h = scan(Abar, Bu, h0, self.scan_method)
        y = (h * Cf).sum(-1) + self.p["Dskip"] * u
        self.cache = (u, z, delta, A, Abar, Bf, Cf, h, h0)
        self.last_state = h[:, -1]
        return y

    def backward(self, dy):
        u, z, delta, A, Abar, Bf, Cf, h, h0 = self.cache
        p = self.p
        du = dy * p["Dskip"]
        g = {"Dskip": (dy * u).sum((0, 1))}
        dCf = dy[..., None] * h
        gh = reverse_scan(Abar, dy[..., None] * Cf, self.scan_method)  # d loss / d h_t
        first = np.zeros_like(h[:, :1]) if h0 is None else h0[:, None]
        hprev = np.concatenate([first, h[:, :-1]], axis=1)
        d_dA = gh * hprev * Abar  # through Abar = exp(delta * A)
        d_delta = (d_dA * A).sum(-1) + (gh * Bf * u[..., None]).sum(-1)
        g["A_log"] = (d_dA * delta[..., None]).sum((0, 1)) * A
        dBf = gh * delta[..., None] * u[..., None]
        du = du + (gh * delta[..., None] * Bf).sum(-1)
        dz = d_delta * sigmoid(z)
        g["b_dt"] = dz.sum((0, 1))
        if self.selective:
            u2 = u.reshape(-1, u.shape[-1])
            g["W_dt"] = u2.T @ dz.reshape(-1, dz.shape[-1])
            du = du + dz @ p["W_dt"].T
            dBt, dCt = dBf.sum(2), dCf.sum(2)
            g["W_B"] = u2.T @ dBt.reshape(-1, self.n)
            g["W_C"] = u2.T @ dCt.reshape(-1, self.n)
            du = du + dBt @ p["W_B"].T + dCt @ p["W_C"].T
        else:
            g["B"] = dBf.sum((0, 1))
            g["C"] = dCf.sum((0, 1))
        self.g = g
        return du

    def step(self, u, h):
        """Constant-memory inference for one token. u (B, D), h (B, D, N) -> (y, h_new)."""
        z, delta, A, Abar, Bf, Cf = self._coeffs(u)
        h = Abar * h + delta[..., None] * Bf * u[..., None]
        y = (h * Cf).sum(-1) + self.p["Dskip"] * u
        return y, h

    def init_state(self, batch):
        return np.zeros((batch, self.d, self.n))

    def kernel(self, length):
        """For the LTI layer: the equivalent convolution kernel K[t, d] (y = K * u + D u)."""
        if self.selective:
            raise ValueError("a selective SSM has no fixed convolution kernel")
        delta = softplus(self.p["b_dt"])  # (D,)
        A = -np.exp(self.p["A_log"])
        t = np.arange(length)[:, None, None]
        pw = np.exp(t * delta[None, :, None] * A[None])  # Abar^t
        return (pw * (delta[:, None] * self.p["B"])[None] * self.p["C"][None]).sum(-1)
