import numpy as np


class SGD:
    def __init__(self, params, lr=0.1, momentum=0.0):
        self.params, self.lr, self.mom = list(params), lr, momentum
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        for p, v in zip(self.params, self.v):
            v *= self.mom
            v += p.grad
            p.data -= self.lr * v


class Adam:
    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = list(params), lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * p.grad
            v *= self.b2; v += (1 - self.b2) * p.grad ** 2
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter): the decay is
    applied to the weights directly, not folded into the gradient."""

    def __init__(self, params, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, b1, b2, eps)
        self.wd = weight_decay

    def step(self):
        if self.wd:
            for p in self.params:
                p.data -= self.lr * self.wd * p.data
        super().step()


def clip_grad_norm(params, max_norm):
    """Scale grads in place so their global L2 norm is <= max_norm. Returns the pre-clip norm."""
    params = list(params)
    total = float(np.sqrt(sum(float((p.grad ** 2).sum()) for p in params)))
    if total > max_norm:
        scale = max_norm / (total + 1e-12)
        for p in params:
            p.grad *= scale
    return total


def cosine_warmup(step, total, base_lr, warmup=0, min_lr=0.0):
    """LR at 1-indexed `step`: linear warmup then cosine decay to min_lr at `total`."""
    if warmup and step <= warmup:
        return base_lr * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * prog))
