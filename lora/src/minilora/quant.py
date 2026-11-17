"""Block-wise weight quantization for a frozen base model (the "Q" in QLoRA).

Three schemes, all absmax-scaled per block of ``block`` consecutive weights:

* ``int8``: symmetric 8-bit integers in [-127, 127].
* ``int4``: symmetric uniform 4-bit integers in [-7, 7].
* ``nf4``:  4-bit NormalFloat, a 16-entry codebook whose levels sit at the quantiles of a
  standard normal. Pretrained weights are roughly Gaussian, so this spends its few levels
  where the weights actually are and beats uniform ``int4`` at the same size.

``QuantizedBase`` swaps a model's linear weights for their dequantized values and keeps the
integer codes, so a ``GPTLoRA`` trained on top sees exactly what a real QLoRA sees: a frozen
low-precision base, full-precision adapters, gradients flowing through the dequantized weights.
Storage numbers (``stored_bytes``) are what the packed codes plus per-block scales would occupy.

``double_quant=True`` adds QLoRA's second trick: the fp32 per-block scales are themselves
quantized to int8 (mean-centred, absmax-scaled in blocks of ``dq_block``, default 256). The
scale overhead drops from 32/block bits per weight (0.5 at block 64) to about 8/block (0.127),
which is what makes small blocks affordable. The weight codes are chosen against the
*dequantized* scales, so the stored representation is self-consistent and what the model sees is
exactly what ``dequantize`` returns.
"""
from __future__ import annotations

import numpy as np

# QLoRA's NormalFloat-4 codebook (Dettmers et al., 2023): 16 levels in [-1, 1], exact zero included.
NF4 = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141391327, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
], dtype=np.float64)

SCHEMES = ("int8", "int4", "nf4")
DEFAULT_TARGETS = ("qkv_W", "proj_W", "fc_W", "out_W")


class QTensor:
    """One quantized array: integer codes (unpacked, uint8/int8), per-block scales, original shape.

    ``scales`` is always the float32 scale each block is dequantized with. With double quantization
    it is derived from ``dq`` (int8 scale codes, fp32 second-level scales, fp32 mean) and only those
    are counted as stored.
    """

    def __init__(self, scheme, block, shape, codes, scales, dq=None):
        self.scheme, self.block, self.shape = scheme, block, tuple(shape)
        self.codes, self.scales, self.dq = codes, scales, dq

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def stored_bytes(self) -> int:
        bits = 8 if self.scheme == "int8" else 4
        code_bytes = int(np.ceil(self.codes.size * bits / 8))
        if self.dq is None:
            return code_bytes + self.scales.size * self.scales.itemsize
        return code_bytes + self.dq["codes"].size + self.dq["scales"].size * 4 + 4  # int8 codes, fp32 scales, fp32 mean

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.stored_bytes / self.size


def _blocks(x: np.ndarray, block: int) -> np.ndarray:
    flat = np.asarray(x, dtype=np.float64).ravel()
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    return flat.reshape(-1, block)


def _dq_encode(scales: np.ndarray, dq_block: int) -> dict:
    """Quantize positive fp32 scales to int8: subtract the mean, then absmax-scale in blocks of ``dq_block``."""
    mean = np.float32(scales.mean())
    c = _blocks(scales - np.float64(mean), dq_block)
    m = np.abs(c).max(axis=1)
    m[m == 0] = 1.0
    d = (m / 127.0).astype(np.float32)
    codes = np.clip(np.rint(c / d.astype(np.float64)[:, None]), -127, 127).astype(np.int8).ravel()[:scales.size]
    return {"codes": codes, "scales": d, "mean": mean, "block": dq_block, "n": scales.size}


def _dq_decode(dq: dict) -> np.ndarray:
    """Effective per-block scales (float32); the single decode path used by quantize, dequantize and load."""
    codes = dq["codes"].astype(np.float64)
    pad = (-codes.size) % dq["block"]
    if pad:
        codes = np.concatenate([codes, np.zeros(pad)])
    vals = codes.reshape(-1, dq["block"]) * dq["scales"].astype(np.float64)[:, None] + np.float64(dq["mean"])
    return np.maximum(vals.ravel()[:dq["n"]], _MIN_SCALE).astype(np.float32)


_MIN_SCALE = 1e-20  # a double-quantized scale that rounds to <= 0 would zero its whole block


def quantize(W: np.ndarray, scheme: str = "nf4", block: int = 64, double_quant: bool = False,
             dq_block: int = 256) -> QTensor:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
    if block < 1:
        raise ValueError("block must be >= 1")
    if double_quant and dq_block < 1:
        raise ValueError("dq_block must be >= 1")
    B = _blocks(W, block)
    absmax = np.abs(B).max(axis=1)
    absmax[absmax == 0] = 1.0  # all-zero block: any scale reproduces zeros
    qmax = 127 if scheme == "int8" else 7
    scales = absmax if scheme == "nf4" else absmax / qmax
    dq = None
    if double_quant:
        dq = _dq_encode(scales, dq_block)
        scales = _dq_decode(dq).astype(np.float64)  # choose weight codes against what will be dequantized
    if scheme == "nf4":
        norm = np.clip(B / scales[:, None], -1.0, 1.0)
        codes = np.abs(norm[..., None] - NF4).argmin(-1).astype(np.uint8)  # nearest level
    else:
        codes = np.clip(np.rint(B / scales[:, None]), -qmax, qmax).astype(np.int8)
    return QTensor(scheme, block, np.shape(W), codes, scales.astype(np.float32), dq)


def dequantize(q: QTensor, dtype=np.float32) -> np.ndarray:
    s = q.scales.astype(np.float64)[:, None]
    vals = NF4[q.codes] * s if q.scheme == "nf4" else q.codes.astype(np.float64) * s
    return vals.ravel()[:q.size].reshape(q.shape).astype(dtype)


def pack4(codes: np.ndarray) -> np.ndarray:
    """Two 4-bit codes per byte (codes are offset to 0..15 for the signed int4 scheme)."""
    flat = codes.ravel().astype(np.int16)
    if flat.size % 2:
        flat = np.append(flat, 0)
    return (flat[0::2] | (flat[1::2] << 4)).astype(np.uint8)


def unpack4(packed: np.ndarray, n: int) -> np.ndarray:
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2], out[1::2] = packed & 0x0F, packed >> 4
    return out[:n]


class QuantizedBase:
    """Replace a model's linear weights with their quantized values, keeping the codes.

    The model's own parameters become the dequantized tensors, so everything downstream
    (forward, ``GPTLoRA``, the hand-written backward) works unchanged. The base must stay
    frozen: ``verify_frozen()`` checks the parameters still equal the dequantized codes.
    ``source`` (optional ``{name: array}``) overrides what gets quantized for those tensors.
    """

    def __init__(self, model, scheme="nf4", block=64, targets=DEFAULT_TARGETS, embeddings=False, source=None,
                 double_quant=False, dq_block=256):
        if scheme not in SCHEMES:
            raise ValueError(f"unknown scheme {scheme!r}; choose from {SCHEMES}")
        self.model, self.scheme, self.block = model, scheme, block
        self.double_quant, self.dq_block = double_quant, dq_block
        names = [n for n in model.params if n.split(".")[-1] in targets]
        if embeddings:
            names += [n for n in ("wte", "wpe", "head_W") if n in model.params]
        if not names:
            raise ValueError("no parameters matched the quantization targets")
        self.names = names
        self.q, self.original_bytes, self.sq_err, self.sq_norm = {}, {}, {}, {}
        for n in names:
            w = model.params[n]
            # ``source`` lets a caller quantize something other than the weight itself (LoftQ quantizes
            # W - adapter); error and size accounting stay relative to the real weight ``w``.
            self.q[n] = quantize(source[n] if source and n in source else w, scheme, block, double_quant, dq_block)
            deq = dequantize(self.q[n], w.dtype)
            self.sq_err[n] = float(((w.astype(np.float64) - deq) ** 2).sum())
            self.sq_norm[n] = float((w.astype(np.float64) ** 2).sum())
            self.original_bytes[n] = w.nbytes
            model.params[n] = deq

    # ---- accounting ----
    def stored_bytes(self) -> int:
        """Bytes of the quantized tensors only (codes + scales)."""
        return sum(q.stored_bytes for q in self.q.values())

    def fp_bytes(self) -> int:
        """Bytes the same tensors take at the model's own dtype."""
        return sum(self.original_bytes.values())

    def bits_per_weight(self) -> float:
        """Average stored bits per quantized weight (codes + scales, nothing else)."""
        return 8.0 * self.stored_bytes() / sum(q.size for q in self.q.values())

    def compression(self) -> float:
        return self.fp_bytes() / self.stored_bytes()

    def model_bytes(self) -> int:
        """Whole-model footprint: quantized tensors packed, everything else left as is."""
        rest = sum(v.nbytes for k, v in self.model.params.items() if k not in self.q)
        return rest + self.stored_bytes()

    def relative_error(self) -> float:
        """sqrt(sum ||W - deq(W)||^2 / sum ||W||^2) over all quantized tensors."""
        return float(np.sqrt(sum(self.sq_err.values()) / max(sum(self.sq_norm.values()), 1e-30)))

    def tensor_errors(self) -> dict:
        return {n: float(np.sqrt(self.sq_err[n] / max(self.sq_norm[n], 1e-30))) for n in self.names}

    # ---- integrity ----
    def verify_frozen(self) -> bool:
        return all(np.array_equal(self.model.params[n], dequantize(q, self.model.params[n].dtype))
                   for n, q in self.q.items())

    # ---- files ----
    def save(self, path):
        """Write codes (nibble-packed for 4-bit) and scales; the rest of the model is not included."""
        arrays = {"__meta__": np.array([self.scheme, str(self.block), str(self.dq_block if self.double_quant else 0)])}
        for n, q in self.q.items():
            c = q.codes if q.scheme == "int8" else pack4(q.codes.astype(np.int16) + (8 if q.scheme == "int4" else 0))
            arrays[f"codes::{n}"] = c
            if q.dq is None:
                arrays[f"scales::{n}"] = q.scales
            else:  # only the int8 scale codes + second-level scales + mean are stored
                arrays[f"dqcodes::{n}"], arrays[f"dqscales::{n}"] = q.dq["codes"], q.dq["scales"]
                arrays[f"dqmean::{n}"] = np.array([q.dq["mean"]], dtype=np.float32)
            arrays[f"shape::{n}"] = np.array(q.shape)
        np.savez(path, **arrays)

    @staticmethod
    def load_tensors(path) -> dict:
        """Read a file written by ``save`` back into ``{name: QTensor}``."""
        out = {}
        with np.load(path) as z:
            meta = z["__meta__"]
            scheme, block = str(meta[0]), int(meta[1])
            dq_block = int(meta[2]) if meta.size > 2 else 0
            for key in z.files:
                if not key.startswith("codes::"):
                    continue
                n = key[7:]
                shape = tuple(int(s) for s in z[f"shape::{n}"])
                dq = None
                if dq_block:
                    dqc = z[f"dqcodes::{n}"]
                    n_blocks = int(np.ceil(int(np.prod(shape)) / block))
                    dq = {"codes": dqc, "scales": z[f"dqscales::{n}"], "mean": np.float32(z[f"dqmean::{n}"][0]),
                          "block": dq_block, "n": n_blocks}
                    scales = _dq_decode(dq)
                else:
                    scales = z[f"scales::{n}"]
                count = scales.size * block
                if scheme == "int8":
                    codes = z[key]
                elif scheme == "int4":
                    codes = (unpack4(z[key], count).astype(np.int16) - 8).astype(np.int8)
                else:
                    codes = unpack4(z[key], count)
                out[n] = QTensor(scheme, block, shape, codes.reshape(-1, block), scales, dq)
        return out
