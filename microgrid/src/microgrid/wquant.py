"""Weight-only quantisation of MiniGPT (simulated: weights are rounded to a b-bit grid and
stored back as float64, so any forward path - autodiff, KV cache - runs unchanged).

Linear weights are (in, out); "per-channel" uses one symmetric scale per output column,
"per-tensor" one scale for the whole matrix. Embeddings use one scale per row (token).
LayerNorm gains/biases and linear biases stay float (they are tiny).
"""
import copy
import numpy as np


def quantize_array(w, bits=8, axis=None):
    """Symmetric round-to-nearest. `axis` = axis reduced when taking max|w| (None = per tensor).
    Returns (integer codes, scale) with w ~= codes * scale."""
    qmax = 2 ** (bits - 1) - 1
    s = np.abs(w).max(axis=axis, keepdims=axis is not None) / qmax
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.rint(w / s), -qmax, qmax).astype(np.int32), s


def fake_quant(w, bits=8, axis=None):
    q, s = quantize_array(w, bits, axis)
    return q * s


def fake_quant_grouped(w, bits=8, group=16):
    """Group-wise symmetric RTN for an (in, out) matrix: each run of `group` consecutive input
    rows within an output column gets its own scale (last group may be shorter)."""
    out = np.empty_like(w, dtype=float)
    for i in range(0, w.shape[0], group):
        out[i:i + group] = fake_quant(w[i:i + group], bits, 0)
    return out


def _linears(model):
    for b in model.blocks:
        yield b.attn.qkv
        yield b.attn.proj
        yield b.fc1
        yield b.fc2
    yield model.head


def quantize_model(model, bits=8, per_channel=True, embeddings=True, group=None):
    """Return a deep copy with weights rounded to `bits` bits; the original is untouched."""
    m = copy.deepcopy(model)
    for l in _linears(m):
        if group:
            l.w.data = fake_quant_grouped(l.w.data, bits, group)
        else:
            l.w.data = fake_quant(l.w.data, bits, 0 if per_channel else None)
    if embeddings:
        m.tok.data = fake_quant(m.tok.data, bits, 1 if per_channel else None)
        m.pos.data = fake_quant(m.pos.data, bits, 1 if per_channel else None)
    return m


def weight_bytes(model, bits=None, per_channel=True, embeddings=True, group=None):
    """Storage estimate in bytes. bits=None -> float64 as held in memory. Otherwise packed
    b-bit codes plus float32 scales for quantised tensors, float32 for the unquantised rest."""
    lin = [l.w.data for l in _linears(model)]
    emb = [model.tok.data, model.pos.data]
    rest = [l.b.data for l in _linears(model)]
    for b in model.blocks:
        rest += [b.ln1.g.data, b.ln1.b.data, b.ln2.g.data, b.ln2.b.data]
    rest += [model.lnf.g.data, model.lnf.b.data]
    if bits is None:
        return sum(a.nbytes for a in lin + emb + rest)
    total = 4 * sum(a.size for a in rest)
    for a, is_emb in [(a, False) for a in lin] + [(a, True) for a in emb]:
        if is_emb and not embeddings:
            total += 4 * a.size
            continue
        nscales = (a.shape[0] if is_emb else a.shape[1]) if per_channel else 1
        if group and not is_emb:
            nscales = int(np.ceil(a.shape[0] / group)) * a.shape[1]
        total += int(np.ceil(a.size * bits / 8)) + 4 * nscales
    return total
