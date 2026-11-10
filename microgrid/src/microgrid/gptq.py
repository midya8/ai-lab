"""GPTQ-style error-compensated rounding for MiniGPT linear layers.

Round-to-nearest (wquant) rounds every weight independently. GPTQ (Frantar et al. 2022) rounds the
input rows of a layer one at a time and pushes each rounding error onto the not-yet-rounded rows,
using the inverse Hessian H = X^T X of the layer's calibration inputs, so the layer *output* on
typical inputs moves less. Weights here are (in, out); scales are per output column, fixed from the
original weights (same grid as per-channel RTN, so only the rounding decisions differ).

Simplification: Hessians are collected in a single fp64 forward pass over calibration windows (not
re-collected after earlier layers are quantised, as full GPTQ does). Embeddings use plain RTN.
"""
import copy
import numpy as np
from . import nn
from .wquant import _linears, quantize_array, fake_quant


def collect_hessians(model, windows):
    """windows: int array (n, T). Returns {id(linear): X^T X} for every linear in `model`."""
    lins = {id(l): l for l in _linears(model)}
    H = {k: 0.0 for k in lins}
    orig = nn.Linear.__call__

    def spy(self, x):
        if id(self) in H:
            a = np.asarray(x.data).reshape(-1, x.data.shape[-1])
            H[id(self)] = H[id(self)] + a.T @ a
        return orig(self, x)

    nn.Linear.__call__ = spy
    try:
        model(np.asarray(windows))
    finally:
        nn.Linear.__call__ = orig
    return H


def gptq_quantize(w, H, bits=4, damp=0.01):
    """Error-compensated rounding of w (in, out) given Hessian H (in, in). Returns float64 weights
    on the per-column symmetric grid."""
    w = w.astype(np.float64).copy()
    _, s = quantize_array(w, bits, axis=0)
    qmax = 2 ** (bits - 1) - 1
    H = H.copy()
    dead = np.diag(H) == 0
    H[dead, dead] = 1.0
    w[dead] = 0.0
    H[np.diag_indices_from(H)] += damp * np.mean(np.diag(H))
    U = np.linalg.cholesky(np.linalg.inv(H)).T  # upper: Hinv = U^T U
    out = np.zeros_like(w)
    for i in range(w.shape[0]):
        q = np.clip(np.rint(w[i] / s[0]), -qmax, qmax) * s[0]
        out[i] = q
        err = (w[i] - q) / U[i, i]
        w[i + 1:] -= np.outer(U[i, i + 1:], err)
    return out


def gptq_model(model, windows, bits=4, damp=0.01, embeddings=True):
    """Deep copy with linears GPTQ-rounded using Hessians from `windows`; embeddings per-row RTN."""
    H = collect_hessians(model, windows)
    m = copy.deepcopy(model)
    for l, l0 in zip(_linears(m), _linears(model)):
        l.w.data = gptq_quantize(l0.w.data, H[id(l0)], bits, damp)
    if embeddings:
        m.tok.data = fake_quant(m.tok.data, bits, 1)
        m.pos.data = fake_quant(m.pos.data, bits, 1)
    return m
