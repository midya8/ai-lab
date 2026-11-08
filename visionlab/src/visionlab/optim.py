"""Optimisers that update parameter arrays in place."""
from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, model, lr=0.05, momentum=0.9, weight_decay=0.0):
        self.params = model.parameters()
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.v = [np.zeros_like(p) for _, p, _ in self.params]

    def step(self):
        for v, (name, p, g) in zip(self.v, self.params):
            grad = g + self.wd * p if name.endswith(".W") else g
            v *= self.momentum
            v -= self.lr * grad
            p += v


class Adam:
    def __init__(self, model, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = model.parameters()
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
        self.m = [np.zeros_like(p) for _, p, _ in self.params]
        self.v = [np.zeros_like(p) for _, p, _ in self.params]
        self.t = 0

    def step(self):
        """AdamW-style: decoupled weight decay on weight matrices only."""
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for m, v, (name, p, g) in zip(self.m, self.v, self.params):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            if self.wd and name.endswith(".W"):
                p -= self.lr * self.wd * p
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
