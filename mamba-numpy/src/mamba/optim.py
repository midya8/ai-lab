from __future__ import annotations

import math

import numpy as np


class AdamW:
    """AdamW with global-norm clipping. Decay applies to weight matrices only (names containing '.W' or 'emb')."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), eps=1e-8, wd=0.01, clip=1.0):
        self.params, self.lr, self.betas, self.eps, self.wd, self.clip = params, lr, betas, eps, wd, clip
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        gn = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = min(1.0, self.clip / (gn + 1e-12)) if self.clip else 1.0
        self.t += 1
        b1, b2 = self.betas
        for k, p in self.params.items():
            g = grads[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            if self.wd and (".W" in k or k == "emb"):
                p *= 1 - lr * self.wd
            p -= lr * mh / (np.sqrt(vh) + self.eps)
        return gn


def cosine_lr(step, total, peak, warmup=50, floor=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))
