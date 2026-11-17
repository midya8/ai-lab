"""A stack of graph layers with optional residual connections and PairNorm."""
from __future__ import annotations

import numpy as np

from .graph import GraphOps
from .layers import DenseLayer, GATLayer, GCNLayer, SAGELayer

KINDS = ("mlp", "gcn", "sage", "gat")


def softmax_xent(logits: np.ndarray, y: np.ndarray, idx: np.ndarray):
    """Mean cross-entropy over rows ``idx``; returns (loss, d loss / d logits)."""
    z = logits[idx]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    m = len(idx)
    loss = float(-np.log(p[np.arange(m), y[idx]] + 1e-300).mean())
    dz = p
    dz[np.arange(m), y[idx]] -= 1.0
    dz /= m
    grad = np.zeros_like(logits)
    grad[idx] = dz
    return loss, grad


def pairnorm(x: np.ndarray, scale: float = 1.0, eps: float = 1e-9):
    """PairNorm (Zhao & Akoglu): centre the rows, then rescale so mean ||x_i||^2 = scale^2."""
    xc = x - x.mean(0, keepdims=True)
    r = np.sqrt((xc * xc).sum() / x.shape[0] + eps)
    return scale * xc / r, (xc, r)


def pairnorm_backward(dy: np.ndarray, cache, scale: float = 1.0):
    xc, r = cache
    n = xc.shape[0]
    dxc = scale * (dy / r - xc * (dy * xc).sum() / (n * r ** 3))
    return dxc - dxc.mean(0, keepdims=True)


class GNN:
    """``depth`` layers of ``kind`` in {mlp, gcn, sage, gat}; ReLU between layers.

    Block l (not last):  h <- relu(norm(layer(h))) [+ h if residual and widths match]
    Last block:          logits = layer(h)
    """

    def __init__(self, kind: str, in_dim: int, hidden: int, out_dim: int, depth: int = 2,
                 seed: int = 0, residual: bool = False, pairnorm_scale: float | None = None,
                 heads: int = 4):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if kind == "gat" and hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        rng = np.random.default_rng(seed)
        self.kind, self.depth = kind, depth
        self.residual = residual
        self.pairnorm_scale = pairnorm_scale
        self.layers = []
        for l in range(depth):
            fi = in_dim if l == 0 else hidden
            last = l == depth - 1
            fo = out_dim if last else hidden
            if kind == "mlp":
                self.layers.append(DenseLayer(fi, fo, rng))
            elif kind == "gcn":
                self.layers.append(GCNLayer(fi, fo, rng))
            elif kind == "sage":
                self.layers.append(SAGELayer(fi, fo, rng))
            else:
                self.layers.append(GATLayer(fi, fo, 1, rng) if last
                                   else GATLayer(fi, hidden // heads, heads, rng))
        self.hidden_states: list[np.ndarray] = []

    # -- parameters -----------------------------------------------------------------
    def named_params(self):
        for i, layer in enumerate(self.layers):
            for k in layer.params:
                yield f"{i}.{k}", layer.params[k], layer.grads[k]

    def n_params(self) -> int:
        return int(sum(p.size for _, p, _ in self.named_params()))

    def state(self):
        return {n: p.copy() for n, p, _ in self.named_params()}

    def load_state(self, state):
        for n, p, _ in self.named_params():
            p[...] = state[n]

    # -- forward / backward ---------------------------------------------------------
    def forward(self, x: np.ndarray, ops: GraphOps) -> np.ndarray:
        h = x
        self._cache = []
        self.hidden_states = []
        for l, layer in enumerate(self.layers):
            z = layer.forward(h, ops)
            if l == self.depth - 1:
                self._cache.append(None)
                return z
            pn = None
            if self.pairnorm_scale is not None:
                z, pn = pairnorm(z, self.pairnorm_scale)
            a = np.maximum(z, 0.0)
            skip = self.residual and l > 0
            out = a + h if skip else a
            self._cache.append((z, pn, skip))
            self.hidden_states.append(out)
            h = out
        raise AssertionError("unreachable")

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        d = self.layers[-1].backward(dlogits)
        for l in range(self.depth - 2, -1, -1):
            z, pn, skip = self._cache[l]
            da = d * (z > 0)
            if pn is not None:
                da = pairnorm_backward(da, pn, self.pairnorm_scale)
            dh = self.layers[l].backward(da)
            d = dh + d if skip else dh
        return d
