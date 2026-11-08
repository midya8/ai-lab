"""White-box adversarial attacks (FGSM, PGD) in raw [0,1] pixel space."""
from __future__ import annotations

import numpy as np

from .model import cross_entropy


def loss_grad(model, x, y):
    """Gradient of the mean CE loss w.r.t. the input pixels (model in eval mode)."""
    was = [l.training for l in model.layers]
    model.eval()
    logits = model.forward(x)
    _, dlogits = cross_entropy(logits, y)
    gx = model.backward(dlogits)
    for layer, t in zip(model.layers, was):
        layer.training = t
    return gx


def fgsm(model, x, y, eps):
    """One signed-gradient step of size eps, clipped to the valid pixel range."""
    return np.clip(x + eps * np.sign(loss_grad(model, x, y)), 0.0, 1.0).astype(x.dtype)


def pgd(model, x, y, eps, steps=10, alpha=None, rng=None):
    """Projected gradient descent inside the L-inf ball of radius eps (random start)."""
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    rng = rng if rng is not None else np.random.default_rng(0)
    adv = np.clip(x + rng.uniform(-eps, eps, x.shape), 0, 1).astype(x.dtype)
    for _ in range(steps):
        adv = adv + alpha * np.sign(loss_grad(model, adv, y))
        adv = np.clip(np.clip(adv, x - eps, x + eps), 0, 1).astype(x.dtype)
    return adv
