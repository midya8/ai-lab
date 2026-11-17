"""Models that decouple propagation from transformation, plus a label-propagation baseline.

* ``APPNP``  (Klicpera et al.): an MLP predicts per-node logits ``Z``, then K steps of
  personalised-PageRank propagation  ``H <- (1-alpha) Ahat H + alpha Z``  smooth them. The
  propagation has *no parameters*, so depth K costs no trainability.
* ``SGC``    (Wu et al.): ``logits = (Ahat^K X) W + b``; the propagated features are computed once.
* ``JKNet``  (Xu et al.): a GCN stack whose output layer reads the *concatenation* of every
  layer's hidden state (jumping knowledge), so shallow, un-smoothed features stay reachable.
* ``label_propagation``: Zhou et al.; uses edges and training labels only, never features.

All three models share ``GNN``'s interface (``forward``, ``backward``, ``named_params``,
``state``, ``load_state``, ``hidden_states``) so ``fit`` trains them unchanged.
"""
from __future__ import annotations

import numpy as np

from .graph import GraphOps
from .layers import DenseLayer, GCNLayer


class _ParamsMixin:
    layers: list

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


def propagate_ppr(z: np.ndarray, ahat: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """K steps of H <- (1-alpha) Ahat H + alpha Z starting from H = Z."""
    h = z
    for _ in range(k):
        h = (1.0 - alpha) * (ahat @ h) + alpha * z
    return h


def propagate_ppr_backward(dh: np.ndarray, ahat: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Adjoint of ``propagate_ppr``: returns d loss / d Z given d loss / d H_K."""
    dz = np.zeros_like(dh)
    d = dh
    for _ in range(k):
        dz += alpha * d
        d = (1.0 - alpha) * (ahat.T @ d)
    return dz + d


class APPNP(_ParamsMixin):
    """MLP (``depth`` dense layers, ReLU between) followed by K PPR propagation steps."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, k: int = 10, alpha: float = 0.1,
                 depth: int = 2, seed: int = 0):
        if k < 0:
            raise ValueError("k must be >= 0")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if depth < 1:
            raise ValueError("depth must be >= 1")
        rng = np.random.default_rng(seed)
        self.k, self.alpha, self.depth = k, alpha, depth
        dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]
        self.layers = [DenseLayer(dims[i], dims[i + 1], rng) for i in range(depth)]
        self.hidden_states: list[np.ndarray] = []

    def forward(self, x: np.ndarray, ops: GraphOps) -> np.ndarray:
        h = x
        self._pre = []
        for l, layer in enumerate(self.layers):
            z = layer.forward(h, ops)
            if l < self.depth - 1:
                self._pre.append(z)
                h = np.maximum(z, 0.0)
            else:
                h = z
        self._ahat = ops.ahat
        out = propagate_ppr(h, ops.ahat, self.k, self.alpha)
        self.hidden_states = [out]
        return out

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        d = propagate_ppr_backward(dlogits, self._ahat, self.k, self.alpha)
        for l in range(self.depth - 1, -1, -1):
            d = self.layers[l].backward(d)
            if l > 0:
                d = d * (self._pre[l - 1] > 0)
        return d


class SGC(_ParamsMixin):
    """logits = (Ahat^K X) W + b. The smoothed features are cached per (x, ops) pair."""

    def __init__(self, in_dim: int, out_dim: int, k: int = 2, seed: int = 0):
        if k < 0:
            raise ValueError("k must be >= 0")
        rng = np.random.default_rng(seed)
        self.k = k
        self.layers = [DenseLayer(in_dim, out_dim, rng)]
        self._cache_key = None
        self._cache_val = None
        self.hidden_states: list[np.ndarray] = []

    def smoothed(self, x: np.ndarray, ops: GraphOps) -> np.ndarray:
        if self._cache_key is None or self._cache_key[0] is not x or self._cache_key[1] is not ops:
            h = x
            for _ in range(self.k):
                h = ops.ahat @ h
            self._cache_key, self._cache_val = (x, ops), h
        return self._cache_val

    def forward(self, x: np.ndarray, ops: GraphOps) -> np.ndarray:
        s = self.smoothed(x, ops)
        self.hidden_states = [s]
        return self.layers[0].forward(s, ops)

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        # gradient w.r.t. the *smoothed* features; the input itself is a constant here
        return self.layers[0].backward(dlogits)


class JKNet(_ParamsMixin):
    """``depth`` GCN layers (ReLU) -> concat of all hidden states -> linear classifier."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 4, seed: int = 0):
        if depth < 1:
            raise ValueError("depth must be >= 1")
        rng = np.random.default_rng(seed)
        self.depth, self.hidden = depth, hidden
        self.layers = [GCNLayer(in_dim if l == 0 else hidden, hidden, rng) for l in range(depth)]
        self.layers.append(DenseLayer(depth * hidden, out_dim, rng))
        self.hidden_states: list[np.ndarray] = []

    def forward(self, x: np.ndarray, ops: GraphOps) -> np.ndarray:
        h = x
        self._pre, outs = [], []
        for layer in self.layers[:-1]:
            z = layer.forward(h, ops)
            self._pre.append(z)
            h = np.maximum(z, 0.0)
            outs.append(h)
        self.hidden_states = outs
        return self.layers[-1].forward(np.concatenate(outs, axis=1), ops)

    def backward(self, dlogits: np.ndarray) -> np.ndarray:
        dcat = self.layers[-1].backward(dlogits)
        parts = np.split(dcat, self.depth, axis=1)
        d_next = np.zeros_like(parts[-1])
        for l in range(self.depth - 1, -1, -1):
            da = (parts[l] + d_next) * (self._pre[l] > 0)
            d_next = self.layers[l].backward(da)
        return d_next


def label_propagation(ops: GraphOps, y: np.ndarray, train: np.ndarray, k: int,
                      alpha: float = 0.9, iters: int = 50) -> np.ndarray:
    """Zhou et al. label spreading: F <- alpha S F + (1-alpha) Y0, with S = ``ops.ahat``.

    Y0 holds one-hot rows for the labelled ``train`` nodes and zeros elsewhere. Returns F
    (n, k); predict with ``F.argmax(1)``. Features are never used. Nodes the labels cannot
    reach keep an all-zero row (argmax then returns class 0: an honest 'no information').
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    y0 = np.zeros((ops.ahat.shape[0], k))
    y0[train, y[train]] = 1.0
    f = y0.copy()
    for _ in range(iters):
        f = alpha * (ops.ahat @ f) + (1.0 - alpha) * y0
    return f
