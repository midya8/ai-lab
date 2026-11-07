"""Adam optimizer with gradient clipping, operating on dicts of arrays."""
import numpy as np


class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), eps=1e-8, clip=1.0):
        self.params, self.lr, self.betas, self.eps, self.clip = params, lr, betas, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        norm = float(np.sqrt(sum((g ** 2).sum() for g in grads.values())))
        scale = min(1.0, self.clip / (norm + 1e-12)) if self.clip else 1.0
        self.t += 1
        b1, b2 = self.betas
        for k, p in self.params.items():
            g = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return norm
