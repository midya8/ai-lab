"""LoRA from scratch in NumPy: low-rank adapters on a 2-layer MLP with manual backprop."""
import numpy as np


class LoRALinear:
    """y = x W + b + (alpha/r) * x A B ; W frozen, A (in,r) and B (r,out) trainable. B starts at 0."""

    def __init__(self, W, b, r=4, alpha=8.0, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W, self.b = W, b
        self.r, self.scale = r, alpha / r
        self.A = rng.normal(0, 1 / np.sqrt(W.shape[0]), (W.shape[0], r))
        self.B = np.zeros((r, W.shape[1]))
        self.merged = False

    def forward(self, x):
        self.x = x
        y = x @ self.W + self.b
        if not self.merged:
            self.xa = x @ self.A
            y = y + self.scale * (self.xa @ self.B)
        return y

    def backward(self, dy):
        assert not self.merged, "cannot train merged layer"
        self.dB = self.scale * self.xa.T @ dy
        dxa = self.scale * dy @ self.B.T
        self.dA = self.x.T @ dxa
        return dy @ self.W.T + dxa @ self.A.T

    def params(self):
        return [(self.A, "dA"), (self.B, "dB")]

    def merge(self):
        if not self.merged:
            self.W = self.W + self.scale * self.A @ self.B
            self.merged = True

    def unmerge(self):
        if self.merged:
            self.W = self.W - self.scale * self.A @ self.B
            self.merged = False

    def n_trainable(self):
        return self.A.size + self.B.size


class MLP:
    def __init__(self, d_in, d_hid, d_out, rng):
        self.W1 = rng.normal(0, 1 / np.sqrt(d_in), (d_in, d_hid)); self.b1 = np.zeros(d_hid)
        self.W2 = rng.normal(0, 1 / np.sqrt(d_hid), (d_hid, d_out)); self.b2 = np.zeros(d_out)
        self.l1 = self.l2 = None

    def attach_lora(self, r=4, alpha=8.0, seed=0):
        rng = np.random.default_rng(seed)
        self.l1 = LoRALinear(self.W1, self.b1, r, alpha, rng)
        self.l2 = LoRALinear(self.W2, self.b2, r, alpha, rng)

    def forward(self, x):
        l1, l2 = self.l1, self.l2
        if l1 is None:
            self.x = x; self.h = np.tanh(x @ self.W1 + self.b1)
            return self.h @ self.W2 + self.b2
        self.h = np.tanh(l1.forward(x))
        return l2.forward(self.h)

    def loss_grad(self, logits, y):
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        d = p.copy(); d[np.arange(n), y] -= 1
        return loss, d / n

    def step_full(self, x, y, lr):
        """Full fine-tuning / pretraining step (updates all weights)."""
        logits = self.forward(x)
        loss, d = self.loss_grad(logits, y)
        dW2 = self.h.T @ d; db2 = d.sum(0)
        dh = (d @ self.W2.T) * (1 - self.h ** 2)
        self.W1 -= lr * (x.T @ dh); self.b1 -= lr * dh.sum(0)
        self.W2 -= lr * dW2; self.b2 -= lr * db2
        return loss

    def step_lora(self, x, y, lr):
        logits = self.forward(x)
        loss, d = self.loss_grad(logits, y)
        dh = self.l2.backward(d) * (1 - self.h ** 2)
        self.l1.backward(dh)
        for l in (self.l1, self.l2):
            for p, g in l.params():
                p -= lr * getattr(l, g)
        return loss

    def merge(self):
        self.l1.merge(); self.l2.merge()
        self.W1, self.W2 = self.l1.W, self.l2.W

    def predict(self, x):
        return self.forward(x).argmax(1)


def make_task(n, d, k, seed, rot=0.0):
    """Gaussian blobs; `rot` perturbs class centers so task B differs from task A."""
    rng = np.random.default_rng(seed)
    base = np.random.default_rng(123).normal(0, 2, (k, d))
    centers = base + rot * np.random.default_rng(456).normal(0, 2, (k, d))
    y = rng.integers(0, k, n)
    return centers[y] + rng.normal(0, 1, (n, d)), y
