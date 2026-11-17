"""GNN layers with hand-written forward and backward passes.

Every layer exposes ``forward(h, ops)``, ``backward(dout) -> dh`` and ``params`` / ``grads``
dicts of arrays (updated in place by the optimiser).
"""
from __future__ import annotations

import numpy as np

from .graph import GraphOps


def glorot(rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=(fan_in, fan_out))


class DenseLayer:
    """Plain linear layer; ignores the graph (the MLP baseline)."""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator):
        self.params = {"W": glorot(rng, fan_in, fan_out), "b": np.zeros(fan_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, h, ops: GraphOps):
        self._h = h
        return h @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"][...] = self._h.T @ dout
        self.grads["b"][...] = dout.sum(0)
        return dout @ self.params["W"].T


class GCNLayer:
    """Kipf & Welling: out = Ahat H W + b, Ahat = D~^-1/2 (A+I) D~^-1/2."""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator):
        self.params = {"W": glorot(rng, fan_in, fan_out), "b": np.zeros(fan_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, h, ops: GraphOps):
        self._h, self._ahat = h, ops.ahat
        return ops.ahat @ (h @ self.params["W"]) + self.params["b"]

    def backward(self, dout):
        dz = self._ahat.T @ dout
        self.grads["W"][...] = self._h.T @ dz
        self.grads["b"][...] = dout.sum(0)
        return dz @ self.params["W"].T


class SAGELayer:
    """GraphSAGE-mean: out = H Ws + mean_neigh(H) Wn + b (self and neighbours kept separate)."""

    def __init__(self, fan_in: int, fan_out: int, rng: np.random.Generator):
        self.params = {
            "Ws": glorot(rng, fan_in, fan_out),
            "Wn": glorot(rng, fan_in, fan_out),
            "b": np.zeros(fan_out),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, h, ops: GraphOps):
        self._h, self._mean = h, ops.mean
        self._nh = ops.mean @ h
        return h @ self.params["Ws"] + self._nh @ self.params["Wn"] + self.params["b"]

    def backward(self, dout):
        self.grads["Ws"][...] = self._h.T @ dout
        self.grads["Wn"][...] = self._nh.T @ dout
        self.grads["b"][...] = dout.sum(0)
        return dout @ self.params["Ws"].T + self._mean.T @ (dout @ self.params["Wn"].T)


class GATLayer:
    """Graph attention (Velickovic et al.), ``heads`` heads of width ``d_head``, concatenated.

    e_ij = LeakyReLU(a_src . z_i + a_dst . z_j) for j in N(i) + {i}; alpha = softmax_j(e_ij);
    out_i = sum_j alpha_ij z_j.  fan_out = heads * d_head.
    """

    def __init__(self, fan_in: int, d_head: int, heads: int, rng: np.random.Generator,
                 slope: float = 0.2):
        self.heads, self.d_head, self.slope = heads, d_head, slope
        self.params = {
            "W": glorot(rng, fan_in, heads * d_head),
            "a_src": rng.normal(size=(heads, d_head)) * np.sqrt(1.0 / d_head),
            "a_dst": rng.normal(size=(heads, d_head)) * np.sqrt(1.0 / d_head),
            "b": np.zeros(heads * d_head),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.attn: np.ndarray | None = None  # (heads, n, n) of the last forward

    def forward(self, h, ops: GraphOps):
        n = h.shape[0]
        H, D = self.heads, self.d_head
        z = (h @ self.params["W"]).reshape(n, H, D)
        u = np.einsum("nhd,hd->hn", z, self.params["a_src"])
        v = np.einsum("nhd,hd->hn", z, self.params["a_dst"])
        s = u[:, :, None] + v[:, None, :]                       # (H, n, n)
        e = np.where(s > 0, s, self.slope * s)
        e = np.where(ops.mask[None], e, -np.inf)
        e = e - e.max(axis=2, keepdims=True)
        w = np.exp(e)
        alpha = w / w.sum(axis=2, keepdims=True)
        zt = z.transpose(1, 0, 2)                                # (H, n, D)
        out = (alpha @ zt).transpose(1, 0, 2).reshape(n, H * D)
        self._h, self._z, self._s, self._alpha = h, z, s, alpha
        self.attn = alpha
        return out + self.params["b"]

    def backward(self, dout):
        n = dout.shape[0]
        H, D = self.heads, self.d_head
        z, s, alpha = self._z, self._s, self._alpha
        do = dout.reshape(n, H, D)
        self.grads["b"][...] = dout.sum(0)
        zt, dot = z.transpose(1, 0, 2), do.transpose(1, 0, 2)  # (H, n, D)
        dalpha = dot @ zt.transpose(0, 2, 1)
        dz = (alpha.transpose(0, 2, 1) @ dot).transpose(1, 0, 2)
        de = alpha * (dalpha - (alpha * dalpha).sum(axis=2, keepdims=True))
        ds = de * np.where(s > 0, 1.0, self.slope)              # masked entries have alpha=0
        du = ds.sum(axis=2)                                      # (H, n)
        dv = ds.sum(axis=1)
        self.grads["a_src"][...] = np.einsum("nhd,hn->hd", z, du)
        self.grads["a_dst"][...] = np.einsum("nhd,hn->hd", z, dv)
        dz = dz + np.einsum("hn,hd->nhd", du, self.params["a_src"]) \
                + np.einsum("hn,hd->nhd", dv, self.params["a_dst"])
        dz = dz.reshape(n, H * D)
        self.grads["W"][...] = self._h.T @ dz
        return dz @ self.params["W"].T
