"""Attribution methods, all hand-built on the model's own backward pass.

* ``saliency``    - |d logit_c / d pixel|
* ``smoothgrad``  - saliency averaged over noisy copies of the input
* ``grad_cam``    - gradient-weighted activation map of the last conv block
* ``occlusion``   - drop in class probability when a patch is blanked
and ``deletion_score`` which tests whether a heat map is *faithful*: delete the
pixels it ranks highest and watch the class probability fall (lower = better).
"""
from __future__ import annotations

import numpy as np

from .layers import Conv2D, ReLU
from .model import softmax


def _target(model, x, cls):
    logits = model.predict_logits(x[None])[0]
    return int(logits.argmax()) if cls is None else int(cls)


def _logit_grad(model, x, cls):
    """Gradient of one logit w.r.t. the input (eval mode); returns (dx, logits)."""
    was = [l.training for l in model.layers]
    model.eval()
    logits = model.forward(x[None])
    g = np.zeros_like(logits)
    g[0, cls] = 1.0
    dx = model.backward(g)[0]
    for layer, t in zip(model.layers, was):
        layer.training = t
    return dx, logits


def normalize01(h):
    h = np.asarray(h, dtype=np.float64)
    lo, hi = h.min(), h.max()
    return np.zeros_like(h) if hi - lo < 1e-12 else (h - lo) / (hi - lo)


def saliency(model, x, cls=None):
    cls = _target(model, x, cls)
    dx, _ = _logit_grad(model, x, cls)
    return np.abs(dx).max(axis=0)


def smoothgrad(model, x, cls=None, n=16, sigma=0.1, seed=0):
    cls = _target(model, x, cls)
    rng = np.random.default_rng(seed)
    acc = np.zeros(x.shape[-2:])
    for _ in range(n):
        dx, _ = _logit_grad(model, (x + rng.normal(0, sigma, x.shape)).astype(x.dtype), cls)
        acc += np.abs(dx).max(axis=0)
    return acc / n


def last_conv_relu(model):
    """Index of the ReLU that follows the last Conv2D (the Grad-CAM target layer)."""
    idx = [i for i, l in enumerate(model.layers) if isinstance(l, Conv2D)]
    if not idx:
        raise ValueError("model has no Conv2D layer")
    for j in range(idx[-1] + 1, len(model.layers)):
        if isinstance(model.layers[j], ReLU):
            return j
    return idx[-1]


def grad_cam(model, x, cls=None):
    """Grad-CAM (Selvaraju et al.) at the last conv block, upsampled to input size."""
    cls = _target(model, x, cls)
    _logit_grad(model, x, cls)
    j = last_conv_relu(model)
    A = model.acts[j][0]            # (K, h, w) activations
    G = model.gouts[j][0]           # d logit / d activation
    alpha = G.mean(axis=(1, 2))     # channel weights
    cam = np.maximum((alpha[:, None, None] * A).sum(axis=0), 0)
    H, W = x.shape[-2:]
    fy, fx = H // cam.shape[0], W // cam.shape[1]
    up = np.kron(cam, np.ones((fy, fx)))
    return _smooth(up)


def _smooth(h):
    """3x3 box blur (edge-padded) to soften the nearest-neighbour upsample blocks."""
    p = np.pad(h, 1, mode="edge")
    return sum(p[i:i + h.shape[0], j:j + h.shape[1]] for i in range(3) for j in range(3)) / 9.0


def occlusion(model, x, cls=None, patch=5, stride=2, fill=0.0):
    """Mean probability drop of `cls` over all occluding windows covering each pixel."""
    cls = _target(model, x, cls)
    H, W = x.shape[-2:]
    base = softmax(model.predict_logits(x[None]))[0, cls]
    boxes, batch = [], []
    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            xo = x.copy()
            xo[:, r:r + patch, c:c + patch] = fill
            boxes.append((r, c))
            batch.append(xo)
    probs = softmax(model.predict_logits(np.stack(batch)))[:, cls]
    heat = np.zeros((H, W))
    cnt = np.zeros((H, W))
    for (r, c), p in zip(boxes, probs):
        heat[r:r + patch, c:c + patch] += base - p
        cnt[r:r + patch, c:c + patch] += 1
    return heat / np.maximum(cnt, 1)


METHODS = {"saliency": saliency, "smoothgrad": smoothgrad, "grad-cam": grad_cam, "occlusion": occlusion}


def deletion_curve(model, x, heat, cls, fractions, fill=0.0):
    """Probability of `cls` after blanking the top-`f` fraction of pixels ranked by `heat`."""
    flat = heat.reshape(-1)
    order = np.argsort(-flat, kind="stable")
    batch = []
    for f in fractions:
        xd = x.copy().reshape(x.shape[0], -1)
        xd[:, order[:int(round(f * flat.size))]] = fill
        batch.append(xd.reshape(x.shape))
    return softmax(model.predict_logits(np.stack(batch)))[:, cls]


def deletion_score(model, X, heat_fn, fractions=None, classes=None, only_correct=None, y=None):
    """Mean area under the deletion curve over images (lower = more faithful map).

    If `y` is given with only_correct=True, only correctly classified images are used.
    """
    fractions = np.linspace(0, 0.4, 9) if fractions is None else np.asarray(fractions)
    preds = model.predict(X)
    curves = []
    for i, x in enumerate(X):
        if only_correct and y is not None and preds[i] != y[i]:
            continue
        cls = int(preds[i]) if classes is None else int(classes[i])
        curves.append(deletion_curve(model, x, heat_fn(x, cls), cls, fractions))
    curves = np.array(curves)
    return float(curves.mean()), curves.mean(axis=0), len(curves)


def random_heat(shape, rng):
    return rng.random(shape)
