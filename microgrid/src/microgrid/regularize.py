"""Dropout (inverted scaling) as an autodiff op."""
import numpy as np
from .tensor import Tensor


def dropout(x, p, rng, training=True):
    """Zero each element with prob p and scale survivors by 1/(1-p).
    Identity when not training or p == 0. The mask is drawn once per call, so
    backward routes gradient only through kept elements."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    if not training or p == 0.0:
        return x
    mask = (rng.random(x.shape) >= p) / (1.0 - p)
    return x * Tensor(mask)


class Dropout:
    def __init__(self, p=0.1, seed=0):
        self.p, self.training = p, True
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        return dropout(x, self.p, self.rng, self.training)

    def train(self): self.training = True; return self
    def eval(self): self.training = False; return self

    def parameters(self):
        return []
