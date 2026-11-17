"""ReLU MLP in the NTK parametrisation, with hand-written per-sample backprop.

Parameters are all N(0, 1); the width scaling lives in the forward pass:

    h_1     = sqrt(sw2_1 / d)       * x   @ W_1^T + sqrt(sb2) * b_1
    h_{l+1} = sqrt(sw2_{l+1} / n_l) * a_l @ W_{l+1}^T + sqrt(sb2) * b_{l+1},   a_l = relu(h_l)

Hidden layers use sw2 = 2 (He, so the pre-activation variance is preserved) and the
scalar output layer uses sw2 = 1.  With this scaling the empirical NTK
Theta(x, x') = sum_theta df(x)/dtheta * df(x')/dtheta converges to a deterministic
kernel as the widths grow (see analytic.py).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

SW2_HIDDEN = 2.0
SW2_OUT = 1.0


@dataclass
class Cache:
    X: np.ndarray
    pre: list      # pre-activations h_1 .. h_{L+1}, each (n, width)
    act: list      # inputs to each linear layer: a_0 = X, a_1 .. a_L


class MLP:
    """Fully connected ReLU net with scalar output, NTK parametrisation."""

    def __init__(self, in_dim: int, width: int, depth: int, sb2: float = 0.1,
                 seed: int = 0, rng: np.random.Generator | None = None):
        if depth < 0:
            raise ValueError("depth (number of hidden layers) must be >= 0")
        rng = rng if rng is not None else np.random.default_rng(seed)
        self.in_dim, self.width, self.depth, self.sb2 = in_dim, width, depth, sb2
        dims = [in_dim] + [width] * depth + [1]
        self.dims = dims
        self.W = [rng.standard_normal((dims[i + 1], dims[i])) for i in range(len(dims) - 1)]
        self.b = [rng.standard_normal(dims[i + 1]) for i in range(len(dims) - 1)]

    # ---- structure -------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return len(self.W)

    def scale(self, layer: int) -> float:
        sw2 = SW2_OUT if layer == self.n_layers - 1 else SW2_HIDDEN
        return float(np.sqrt(sw2 / self.dims[layer]))

    @property
    def sb(self) -> float:
        return float(np.sqrt(self.sb2))

    def copy(self) -> "MLP":
        new = MLP.__new__(MLP)
        new.in_dim, new.width, new.depth, new.sb2 = self.in_dim, self.width, self.depth, self.sb2
        new.dims = list(self.dims)
        new.W = [w.copy() for w in self.W]
        new.b = [b.copy() for b in self.b]
        return new

    def n_params(self) -> int:
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def get_flat(self) -> np.ndarray:
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(self.W, self.b)])

    def set_flat(self, theta: np.ndarray) -> None:
        theta = np.asarray(theta, dtype=float)
        if theta.size != self.n_params():
            raise ValueError("wrong parameter count")
        i = 0
        for k in range(self.n_layers):
            n = self.W[k].size
            self.W[k] = theta[i:i + n].reshape(self.W[k].shape).copy()
            i += n
            m = self.b[k].size
            self.b[k] = theta[i:i + m].copy()
            i += m

    # ---- forward ---------------------------------------------------------
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, Cache]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        a, pre, act = X, [], []
        for k in range(self.n_layers):
            act.append(a)
            h = self.scale(k) * (a @ self.W[k].T) + self.sb * self.b[k]
            pre.append(h)
            a = np.maximum(h, 0.0) if k < self.n_layers - 1 else h
        return pre[-1][:, 0], Cache(X, pre, act)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    # ---- backward --------------------------------------------------------
    def _deltas(self, cache: Cache, gout: np.ndarray) -> list:
        """g_l = d(sum_i gout_i f_i)/d h_l, per sample: list of (n, width_l) arrays."""
        L = self.n_layers
        g = [None] * L
        g[L - 1] = np.asarray(gout, dtype=float).reshape(-1, 1)
        for k in range(L - 2, -1, -1):
            ga = g[k + 1] @ (self.scale(k + 1) * self.W[k + 1])
            g[k] = ga * (cache.pre[k] > 0)
        return g

    def grads(self, cache: Cache, gout: np.ndarray) -> tuple[list, list]:
        """Gradient of sum_i gout_i f(x_i) wrt (W_k, b_k)."""
        g = self._deltas(cache, gout)
        dW = [self.scale(k) * (g[k].T @ cache.act[k]) for k in range(self.n_layers)]
        db = [self.sb * g[k].sum(axis=0) for k in range(self.n_layers)]
        return dW, db

    def grads_flat(self, cache: Cache, gout: np.ndarray) -> np.ndarray:
        dW, db = self.grads(cache, gout)
        return np.concatenate([np.concatenate([w.ravel(), b]) for w, b in zip(dW, db)])

    # ---- Jacobian and empirical NTK -------------------------------------
    def jacobian(self, X: np.ndarray) -> np.ndarray:
        """(n, P) matrix of df(x_i)/dtheta in get_flat() order (explicit, for small nets)."""
        f, cache = self.forward(X)
        g = self._deltas(cache, np.ones(len(f)))
        blocks = []
        for k in range(self.n_layers):
            jw = self.scale(k) * g[k][:, :, None] * cache.act[k][:, None, :]
            blocks.append(jw.reshape(len(f), -1))
            blocks.append(self.sb * g[k])
        return np.concatenate(blocks, axis=1)

    def ntk(self, X1: np.ndarray, X2: np.ndarray | None = None) -> np.ndarray:
        """Empirical NTK without forming the Jacobian: sum over layers of
        (g g'^T) * (a a'^T) * scale^2 plus the bias terms."""
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        f1, c1 = self.forward(X1)
        g1 = self._deltas(c1, np.ones(len(f1)))
        if X2 is None:
            c2, g2 = c1, g1
        else:
            X2 = np.atleast_2d(np.asarray(X2, dtype=float))
            f2, c2 = self.forward(X2)
            g2 = self._deltas(c2, np.ones(len(f2)))
        K = np.zeros((c1.X.shape[0], c2.X.shape[0]))
        for k in range(self.n_layers):
            gg = g1[k] @ g2[k].T
            aa = c1.act[k] @ c2.act[k].T
            K += (self.scale(k) ** 2) * gg * aa + self.sb2 * gg
        return K
