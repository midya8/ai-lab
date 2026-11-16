"""Low-bit KV-cache quantisation: symmetric, grouped along the head dimension, bit-packed.

Each (token, kv-head) vector of length ``d`` is cut into groups of ``group_size`` channels; each
group gets one scale ``amax / qmax`` (stored in float16, rounded *up* so no value ever clips) and
integer codes in ``[-qmax, qmax]``, ``qmax = 2**(bits-1) - 1``. Codes are packed ``8 // bits`` per
byte so the byte counts reported by the cache are the real storage, not a notional one.

Because the scale depends only on one token's vector, appending a token never touches the
quantised values of earlier tokens - which is what a paged cache needs.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_BITS = (2, 4, 8)


def qmax(bits: int) -> int:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    return 2 ** (bits - 1) - 1


def _groups(d: int, group_size):
    g = d if group_size is None else group_size
    if g <= 0 or d % g:
        raise ValueError(f"group_size {g} must divide head_dim {d}")
    return g, d // g


def quantize(x, bits: int, group_size=None, scale_dtype=np.float16):
    """x (..., d) -> (codes int8 (..., d), scales (..., d // group_size)).

    The scale is rounded up to ``scale_dtype`` so ``|x / scale| <= qmax`` and the reconstruction
    error of every element is at most ``scale / 2`` (tested)."""
    x = np.asarray(x, dtype=np.float64)
    g, ng = _groups(x.shape[-1], group_size)
    xg = x.reshape(*x.shape[:-1], ng, g)
    qm = qmax(bits)
    amax = np.abs(xg).max(axis=-1)
    s = amax / qm
    with np.errstate(over="ignore"):
        s_lo = s.astype(scale_dtype)
    s_q = np.where(s_lo.astype(np.float64) < s, np.nextafter(s_lo, scale_dtype(np.inf)), s_lo)
    s_q = np.where(s == 0, scale_dtype(1), s_q).astype(scale_dtype)
    if not np.all(np.isfinite(s_q)):
        raise OverflowError("scale overflows scale_dtype; use float32 scales")
    codes = np.rint(xg / s_q.astype(np.float64)[..., None])
    codes = np.clip(codes, -qm, qm).astype(np.int8).reshape(x.shape)
    return codes, s_q


def dequantize(codes, scales, group_size=None, dtype=np.float64):
    codes = np.asarray(codes)
    g, ng = _groups(codes.shape[-1], group_size)
    xg = codes.reshape(*codes.shape[:-1], ng, g).astype(dtype) * np.asarray(scales).astype(dtype)[..., None]
    return xg.reshape(codes.shape)


def pack(codes, bits: int):
    """int8 codes (..., d) in [-qmax, qmax] -> uint8 (..., d * bits / 8)."""
    qm = qmax(bits)
    per = 8 // bits
    codes = np.asarray(codes)
    d = codes.shape[-1]
    if d % per:
        raise ValueError(f"head_dim {d} must be a multiple of {per} for {bits}-bit packing")
    u = (codes.astype(np.int16) + qm).astype(np.uint8)
    u = u.reshape(*codes.shape[:-1], d // per, per)
    out = np.zeros(u.shape[:-1], dtype=np.uint8)
    for j in range(per):
        out |= (u[..., j] << (bits * j)).astype(np.uint8)
    return out


def unpack(packed, bits: int, d: int):
    qm = qmax(bits)
    per = 8 // bits
    packed = np.asarray(packed, dtype=np.uint8)
    mask = (1 << bits) - 1
    cols = [((packed >> (bits * j)) & mask) for j in range(per)]
    u = np.stack(cols, axis=-1).reshape(*packed.shape[:-1], d)
    return (u.astype(np.int16) - qm).astype(np.int8)


def bytes_per_vector(d: int, bits: int, group_size=None, scale_dtype=np.float16) -> int:
    """Real storage of one quantised (token, head) vector: packed codes + its group scales."""
    g, ng = _groups(d, group_size)
    return d * bits // 8 + ng * np.dtype(scale_dtype).itemsize
