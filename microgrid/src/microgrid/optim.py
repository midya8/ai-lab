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
