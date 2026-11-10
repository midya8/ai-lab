import numpy as np
from .tensor import Tensor


class Linear:
    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng(0)
        self.w = Tensor(rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)))
        self.b = Tensor(np.zeros(n_out))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.layers = [Linear(a, b, rng) for a, b in zip(sizes, sizes[1:])]

    def __call__(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = x.tanh()
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]


def mse_loss(pred, target):
    d = pred - Tensor._wrap(target)
    return (d * d).mean()


def softmax_cross_entropy(logits, labels):
    lp = logits.log_softmax(axis=-1)
    onehot = np.zeros(lp.shape)
    onehot[np.arange(len(labels)), labels] = 1.0
    return -(lp * onehot).sum() * (1.0 / len(labels))
