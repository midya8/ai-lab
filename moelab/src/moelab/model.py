"""Linear -> (MoE | dense FFN) -> Linear regression net with MSE loss."""

from __future__ import annotations

import numpy as np


class Net:
    def __init__(self, d_in: int, d_out: int, layer, d: int, seed: int = 0):
        rng = np.random.default_rng(seed + 100)
        self.W_in = rng.normal(size=(d_in, d)) * np.sqrt(1.0 / d_in)
        self.b_in = np.zeros(d)
        self.layer = layer
        self.W_out = rng.normal(size=(d, d_out)) * np.sqrt(1.0 / d)
        self.b_out = np.zeros(d_out)

    def params(self) -> dict:
        p = {"W_in": self.W_in, "b_in": self.b_in, "W_out": self.W_out, "b_out": self.b_out}
        for k, v in self.layer.params().items():
            p["layer." + k] = v
        return p

    def n_params(self) -> int:
        return sum(v.size for v in self.params().values())

    def forward(self, x: np.ndarray):
        h = x @ self.W_in + self.b_in
        m, lc = self.layer.forward(h)
        y = m @ self.W_out + self.b_out
        return y, (x, h, m, lc)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if hasattr(self.layer, "train_mode"):
            old, self.layer.train_mode = self.layer.train_mode, False
            try:
                return self.forward(x)[0]
            finally:
                self.layer.train_mode = old
        return self.forward(x)[0]

    def loss_and_grads(self, x: np.ndarray, y: np.ndarray):
        pred, (xx, h, m, lc) = self.forward(x)
        n = x.shape[0]
        task = float(np.mean((pred - y) ** 2))
        loss = task + self.layer.aux_loss
        dy = 2.0 * (pred - y) / (n * y.shape[1])
        grads = {"W_out": m.T @ dy, "b_out": dy.sum(0)}
        dm = dy @ self.W_out.T
        dh, lg = self.layer.backward(dm, lc)
        for k, v in lg.items():
            grads["layer." + k] = v
        grads["W_in"] = xx.T @ dh
        grads["b_in"] = dh.sum(0)
        return loss, task, grads
