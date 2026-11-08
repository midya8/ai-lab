"""LoftQ-style initialisation: choose the adapter so that ``Q + s*A@B`` approximates ``W``.

Plain QLoRA quantizes ``W`` to ``Q`` and starts the adapter at zero, so training begins from the
*damaged* model ``Q``. LoftQ (Li et al., 2023) instead alternates

    Q  <- quantize(W - s*A@B)           # quantize what the adapter does not already cover
    A,B <- rank-r SVD of (W - dequant(Q)) # let the adapter absorb the quantization error

so the adapted model starts much closer to the original fp32 weights.

Facts the tests pin down:

* ``iters=1`` is the "SVD init" (quantize W, fit the best rank-r correction to the residual). By
  Eckart-Young its error can never exceed plain quantization's, per tensor.
* Every adapted tensor's final ``Q`` is the quantization of the ``W - s*A@B`` of the *previous*
  iteration, and the returned adapter is the SVD of the residual of that very ``Q``. They are a pair.
* Tensors that quantize but have no adapter (e.g. ``fc``/``out`` when only ``qkv``/``proj`` are
  targeted) are quantized plainly, exactly as ``QuantizedBase`` would.

Unlike zero-init, a LoftQ adapter is part of the approximation of ``W``: "adapter off" is the plain
quantized base, not the original model, and the adapted model is *not* identical to the quantized
base at step 0 (it is closer to fp32).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quant import DEFAULT_TARGETS, QuantizedBase, dequantize, quantize


@dataclass
class LoftQResult:
    qbase: QuantizedBase
    iters: int
    baseline_error: float              # relative error of plain quantization on the adapted tensors
    errors: list = field(default_factory=list)   # after each iteration: ||W - Q - s*A@B|| / ||W||
    tensor_errors: dict = field(default_factory=dict)  # final per-tensor relative error

    @property
    def final_error(self) -> float:
        return self.errors[-1]


def _svd_adapter(R, r, scale):
    """Rank-r factors with ``scale * A @ B`` equal to the best rank-r approximation of ``R``."""
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    k = min(r, S.size)
    root = np.sqrt(S[:k]) / np.sqrt(scale)
    A = np.zeros((R.shape[0], r))
    B = np.zeros((r, R.shape[1]))
    A[:, :k] = U[:, :k] * root
    B[:k, :] = root[:, None] * Vt[:k]
    return A, B


def loftq_init(lora, scheme="nf4", block=64, iters=5, targets=DEFAULT_TARGETS) -> LoftQResult:
    """Quantize ``lora.model`` and initialise ``lora``'s adapters LoftQ-style, in place.

    The model's linear weights are replaced by their dequantized values (a ``QuantizedBase``, returned
    in the result) and ``lora.A`` / ``lora.B`` are overwritten. The model must still hold its fp32 weights.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    uncovered = [n for n in lora.names if n.split(".")[-1] not in targets]
    if uncovered:
        raise ValueError(f"adapted tensors {uncovered} are not among the quantization targets")
    model, s = lora.model, lora.scale
    W = {n: model.params[n].astype(np.float64) for n in lora.names}
    delta = {n: np.zeros_like(W[n]) for n in lora.names}
    A, B, source = {}, {}, {}
    norm2 = sum(float((w ** 2).sum()) for w in W.values())
    baseline = np.sqrt(sum(float(((W[n] - dequantize(quantize(W[n], scheme, block), np.float64)) ** 2).sum())
                           for n in lora.names) / norm2)
    errors = []
    for _ in range(iters):
        err2 = 0.0
        for n in lora.names:
            source[n] = W[n] - delta[n]
            deq = dequantize(quantize(source[n], scheme, block), np.float64)
            A[n], B[n] = _svd_adapter(W[n] - deq, lora.r, s)
            delta[n] = s * A[n] @ B[n]
            err2 += float(((W[n] - deq - delta[n]) ** 2).sum())
        errors.append(float(np.sqrt(err2 / norm2)))

    qb = QuantizedBase(model, scheme, block, targets=targets, source=source)  # quantizes exactly `source`
    for n in lora.names:
        lora.A[n][...] = A[n]
        lora.B[n][...] = B[n]
    tensor_errors = {n: float(np.linalg.norm(W[n] - model.params[n].astype(np.float64) - delta[n])
                              / np.linalg.norm(W[n])) for n in lora.names}
    return LoftQResult(qb, iters, float(baseline), errors, tensor_errors)
