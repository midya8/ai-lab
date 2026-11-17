"""Certified L-inf robustness by interval bound propagation (IBP), forward and backward.

An input box [x-eps, x+eps] (clipped to the valid pixel range [0, 1]) is pushed through the
network as a centre ``mu`` and a non-negative radius ``r`` per activation:

* affine layers (Conv2D, Dense):  mu' = W mu + b,   r' = |W| r      (exact for a box, per layer)
* fixed per-channel affine maps (Normalize, eval-mode BatchNorm2D): scaled by the slope / its absolute value
* ReLU and 2x2 max-pool are monotone, so they act on the lower and upper bound separately
* Flatten / global average pool are linear; Dropout is the identity at inference

The result is a *sound* but loose enclosure of every logit vector the attacker can produce:
a prediction is **certified** when the lower bound of ``logit_y - logit_j`` is positive for every
other class ``j``. Certified accuracy is therefore a rigorous lower bound on accuracy under any
L-inf attack of that size, including attacks nobody has invented yet. (Soundness is for exact
arithmetic: float32 rounding is ignored; the test-suite checks the bounds in float64.)

The backward pass (``ibp_loss_backward``) lets the same bounds be trained through (Gowal et al.,
2018): the loss is cross-entropy on the worst-case logits (true class at its lower bound, every
other class at its upper bound). Models with BatchNorm2D can be certified but not IBP-trained
here, because batch-statistics normalisation of a box is a different problem.
"""
from __future__ import annotations

import copy

import numpy as np

from .layers import BatchNorm2D, Conv2D, Dense, Dropout, Flatten, GlobalAvgPool, MaxPool2D, Normalize, ReLU
from .model import cross_entropy


# --------------------------------------------------------------------------- per-layer interval ops
def _abs_twin(layer):
    """A copy of an affine layer with |W| and no bias (shares nothing it writes to)."""
    t = copy.copy(layer)
    t.W = np.abs(layer.W)
    t.b = np.zeros_like(layer.b)
    t.dW = np.zeros_like(t.W)
    t.db = np.zeros_like(t.b)
    return t


class _AffineOp:
    """Conv2D / Dense: centre through (W, b), radius through (|W|, 0)."""

    def __init__(self, layer):
        self.layer = copy.copy(layer)          # shares W, b, dW, db with the model; own forward cache
        self.twin = _abs_twin(layer)
        self.sign = np.sign(layer.W)

    def forward(self, mu, r):
        return self.layer.forward(mu), self.twin.forward(r)

    def backward(self, gmu, gr):
        gm = self.layer.backward(gmu)           # writes model dW / db
        gr_in = self.twin.backward(gr)          # writes twin dW
        self.layer.dW += self.twin.dW * self.sign
        return gm, gr_in


class _ScaleOp:
    """x -> (x - shift) * scale per channel (or scalar): radius scales by |scale|."""

    def __init__(self, scale, shift):
        self.scale, self.shift = scale, shift

    def forward(self, mu, r):
        return (mu - self.shift) * self.scale, r * np.abs(self.scale)

    def backward(self, gmu, gr):
        return gmu * self.scale, gr * np.abs(self.scale)


class _ReluOp:
    def forward(self, mu, r):
        lo, hi = mu - r, mu + r
        self.ml, self.mh = lo > 0, hi > 0
        lo, hi = lo * self.ml, hi * self.mh
        return (lo + hi) / 2, (hi - lo) / 2

    def backward(self, gmu, gr):
        glo, ghi = (gmu - gr) / 2 * self.ml, (gmu + gr) / 2 * self.mh
        return glo + ghi, ghi - glo


class _PoolOp:
    def __init__(self):
        self.lo_pool, self.hi_pool = MaxPool2D(), MaxPool2D()

    def forward(self, mu, r):
        lo, hi = self.lo_pool.forward(mu - r), self.hi_pool.forward(mu + r)
        return (lo + hi) / 2, (hi - lo) / 2

    def backward(self, gmu, gr):
        glo = self.lo_pool.backward((gmu - gr) / 2)
        ghi = self.hi_pool.backward((gmu + gr) / 2)
        return glo + ghi, ghi - glo


class _LinearShapeOp:
    """Flatten / global average pool: a linear map, so it is applied to centre and radius alike."""

    def __init__(self, layer):
        self.mu_layer, self.r_layer = copy.copy(layer), copy.copy(layer)

    def forward(self, mu, r):
        return self.mu_layer.forward(mu), self.r_layer.forward(r)

    def backward(self, gmu, gr):
        return self.mu_layer.backward(gmu), self.r_layer.backward(gr)


class _IdentityOp:
    def forward(self, mu, r):
        return mu, r

    def backward(self, gmu, gr):
        return gmu, gr


class _BatchNormEvalOp:
    """Eval-mode BatchNorm2D is a per-channel affine map. Certification only (no backward)."""

    def __init__(self, bn):
        s = bn.gamma / np.sqrt(bn.running_var + bn.eps)
        self.s = s[None, :, None, None]
        self.t = (bn.beta - s * bn.running_mean)[None, :, None, None]

    def forward(self, mu, r):
        return mu * self.s + self.t, r * np.abs(self.s)

    def backward(self, gmu, gr):
        raise NotImplementedError("IBP training through BatchNorm2D is not supported")


def _make_op(layer):
    if isinstance(layer, (Conv2D, Dense)):
        return _AffineOp(layer)
    if isinstance(layer, Normalize):
        return _ScaleOp(1.0 / layer.std, layer.mean)
    if isinstance(layer, ReLU):
        return _ReluOp()
    if isinstance(layer, MaxPool2D):
        return _PoolOp()
    if isinstance(layer, (Flatten, GlobalAvgPool)):
        return _LinearShapeOp(layer)
    if isinstance(layer, Dropout):
        return _IdentityOp()
    if isinstance(layer, BatchNorm2D):
        return _BatchNormEvalOp(layer)
    raise TypeError(f"no interval rule for {type(layer).__name__}")


# --------------------------------------------------------------------------- propagation
def input_box(x, eps):
    """Centre and radius of the L-inf ball of radius eps around x, clipped to [0, 1].

    ``eps`` is a scalar or one value per image (shape ``(n,)``).
    """
    eps = np.asarray(eps, dtype=x.dtype)
    if eps.ndim == 1:
        eps = eps.reshape(-1, *([1] * (x.ndim - 1)))
    lo, hi = np.clip(x - eps, 0.0, 1.0), np.clip(x + eps, 0.0, 1.0)
    return (lo + hi) / 2, (hi - lo) / 2


def propagate(model, x, eps, upto=None):
    """Push the box through layers[:upto]; returns (mu, r, ops)."""
    x = np.asarray(x, dtype=model.dtype)
    mu, r = input_box(x, eps)
    ops = []
    for layer in model.layers[:upto]:
        op = _make_op(layer)
        mu, r = op.forward(mu, r)
        ops.append(op)
    return mu, r, ops


def logit_bounds(model, x, eps, batch=256):
    """Sound (lower, upper) bounds on every logit for any input within eps of x."""
    los, his = [], []
    for s in range(0, len(x), batch):
        mu, r, _ = propagate(model, x[s:s + batch], eps)
        los.append(mu - r)
        his.append(mu + r)
    return np.concatenate(los), np.concatenate(his)


def _last_dense(model):
    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], Dense):
            return i
    raise ValueError("model has no Dense output layer")


def margin_lower_bounds(model, x, y, eps, batch=256):
    """Lower bound of ``logit_y - logit_j`` for every class j (``+inf`` where j == y).
    ``eps`` may be a scalar or one radius per image.

    Tighter than subtracting the logit intervals: the final linear layer is applied to the
    *difference* of two logits, so the shared dependence on the last hidden layer cancels.
    """
    k = _last_dense(model)
    dense = model.layers[k]
    out = []
    for s in range(0, len(x), batch):
        xb, yb = x[s:s + batch], y[s:s + batch]
        eb = eps[s:s + batch] if np.ndim(eps) else eps
        mu, r, _ = propagate(model, xb, eb, upto=k)
        if mu.ndim > 2:
            mu, r = mu.reshape(len(mu), -1), r.reshape(len(r), -1)
        d = dense.W[:, yb].T[:, :, None] - dense.W[None, :, :]   # (n, in, K)
        centre = np.einsum("ni,nik->nk", mu, d)
        spread = np.einsum("ni,nik->nk", r, np.abs(d))
        m = centre - spread + (dense.b[yb][:, None] - dense.b[None, :])
        m[np.arange(len(yb)), yb] = np.inf
        out.append(m)
    return np.concatenate(out)


def certified_mask(model, x, y, eps, method="margin"):
    """Boolean per image: provably classified as y for every perturbation within eps.

    ``method="naive"`` compares logit intervals (looser); ``"margin"`` bounds each margin directly.
    """
    if eps == 0:
        return model.predict(x) == y
    if method == "margin":
        return margin_lower_bounds(model, x, y, eps).min(axis=1) > 0
    if method == "naive":
        lo, hi = logit_bounds(model, x, eps)
        hi = hi.copy()
        hi[np.arange(len(y)), y] = -np.inf
        return lo[np.arange(len(y)), y] > hi.max(axis=1)
    raise ValueError("method must be 'margin' or 'naive'")


def certified_accuracy(model, x, y, eps, method="margin"):
    return float(certified_mask(model, x, y, eps, method).mean())


def certified_report(model, x, y, eps=(0.0, 0.01, 0.02, 0.05, 0.1), method="margin"):
    return {e: certified_accuracy(model, x, y, e, method) for e in eps}


def certified_radius(model, x, y, hi=0.5, iters=14):
    """Largest eps (to a resolution of ``hi / 2**iters``) at which each image is certified, by bisection.

    Certification is monotone in eps (a larger box contains a smaller one), so bisection is exact up
    to that resolution. Images the model already misclassifies get 0; images still certified at
    ``hi`` report ``hi`` (so the value is capped).
    """
    lo_r = np.zeros(len(x))
    hi_r = np.full(len(x), float(hi))
    wrong = model.predict(x) != y
    hi_r[wrong] = 0.0
    for _ in range(iters):
        mid = (lo_r + hi_r) / 2
        cert = margin_lower_bounds(model, x, y, mid).min(axis=1) > 0
        cert &= ~wrong
        lo_r = np.where(cert, mid, lo_r)
        hi_r = np.where(cert, hi_r, mid)
    return lo_r


# --------------------------------------------------------------------------- IBP training
def ibp_loss_backward(model, x, y, eps, scale=1.0, label_smoothing=0.0):
    """Worst-case cross-entropy over the eps-box and its gradient w.r.t. every parameter.

    Gradients are *written* into the layers' grad buffers (like ``model.backward``), multiplied by
    ``scale``. Returns ``(loss, worst_case_logits)``. The true class is taken at its lower bound and every
    other class at its upper bound, which is the worst case for softmax cross-entropy (the logit intervals
    are independent here, so this is a bound on the worst case, not always an attainable one).
    """
    mu, r, ops = propagate(model, x, eps)
    n, k = mu.shape
    onehot = np.zeros((n, k), dtype=bool)
    onehot[np.arange(n), y] = True
    z = np.where(onehot, mu - r, mu + r)
    loss, dz = cross_entropy(z, y, label_smoothing)
    dz = dz * scale
    gmu, gr = dz, dz * np.where(onehot, -1.0, 1.0).astype(dz.dtype)
    for op in reversed(ops):
        gmu, gr = op.backward(gmu, gr)
    return loss, z
