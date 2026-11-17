"""Naive (materialise the full score matrix) attention: the ground truth for everything else.

Shapes: q (H, nq, d); k, v (Hkv, nk, d) with H a multiple of Hkv (grouped-query attention;
Hkv == 1 is multi-query attention, Hkv == H is ordinary multi-head attention).
"""
from __future__ import annotations

import numpy as np

from .masks import allowed_mask, alibi_bias


def _group(q, k):
    H, Hkv = q.shape[0], k.shape[0]
    if H % Hkv:
        raise ValueError(f"{H} query heads not divisible by {Hkv} kv heads")
    return H // Hkv


def resolve_offset(q_offset, nq, nk):
    return nk - nq if q_offset is None else q_offset


def naive_attention(q, k, v, *, causal=False, window=None, alibi_slopes=None, q_offset=None,
                    scale=None):
    """Return (out, lse, probs); lse is the log-sum-exp of each row's visible scores."""
    q, k, v = (np.asarray(x, dtype=np.float64) for x in (q, k, v))
    H, nq, d = q.shape
    nk = k.shape[1]
    g = _group(q, k)
    off = resolve_offset(q_offset, nq, nk)
    scale = d ** -0.5 if scale is None else scale
    mask = allowed_mask(nq, nk, off, causal, window)
    out = np.zeros((H, nq, d))
    lse = np.full((H, nq), -np.inf)
    probs = np.zeros((H, nq, nk))
    for h in range(H):
        s = scale * q[h] @ k[h // g].T
        if alibi_slopes is not None:
            s = s + alibi_bias(alibi_slopes[h], nq, nk, off)
        s = np.where(mask, s, -np.inf)
        m = s.max(axis=1)
        ms = np.where(np.isfinite(m), m, 0.0)
        e = np.exp(s - ms[:, None])
        l = e.sum(axis=1)
        ok = l > 0
        p = np.where(ok[:, None], e / np.where(ok, l, 1.0)[:, None], 0.0)
        probs[h] = p
        out[h] = p @ v[h // g]
        lse[h] = np.where(ok, ms + np.log(np.where(ok, l, 1.0)), -np.inf)
    return out, lse, probs


def naive_backward(q, k, v, dout, *, causal=False, window=None, alibi_slopes=None, q_offset=None,
                   scale=None):
    """Analytic gradients (dq, dk, dv) of sum(dout * attention(q, k, v)) w.r.t. q, k, v."""
    q, k, v, dout = (np.asarray(x, dtype=np.float64) for x in (q, k, v, dout))
    H, nq, d = q.shape
    g = _group(q, k)
    scale = d ** -0.5 if scale is None else scale
    _, _, probs = naive_attention(q, k, v, causal=causal, window=window, alibi_slopes=alibi_slopes,
                                  q_offset=q_offset, scale=scale)
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    for h in range(H):
        p = probs[h]
        dv[h // g] += p.T @ dout[h]
        dp = dout[h] @ v[h // g].T
        ds = p * (dp - (dp * p).sum(axis=1, keepdims=True))
        dq[h] = scale * ds @ k[h // g]
        dk[h // g] += scale * ds.T @ q[h]
    return dq, dk, dv


def naive_traffic(nq: int, nk: int, d: int, heads: int = 1) -> dict:
    """Elements moved to/from slow memory by a kernel that materialises S and P (per the
    standard 3-kernel implementation: S = QK^T, P = softmax(S), O = PV)."""
    reads = nq * d + nk * d + nq * nk + nq * nk + nk * d
    writes = nq * nk + nq * nk + nq * d
    return {"reads": heads * reads, "writes": heads * writes, "total": heads * (reads + writes),
            "peak_intermediate": heads * nq * nk}
