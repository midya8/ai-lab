"""Tiled attention with an online softmax (the FlashAttention algorithm) in NumPy.

The score matrix is never materialised: queries and keys are processed in (block_q x block_k)
tiles, a running row maximum ``m`` and denominator ``l`` are rescaled as each key tile arrives,
and tiles that are completely masked (causal / sliding window) are skipped without being
computed. The backward pass stores only the log-sum-exp per row and recomputes the probabilities
tile by tile (FlashAttention-2 style).

Every kernel reports a :class:`Stats` so the I/O and compute claims are measured, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .masks import FULL, PARTIAL, SKIP, alibi_bias, allowed_mask, tile_status
from .reference import _group, resolve_offset


@dataclass
class Stats:
    tiles_total: int = 0
    tiles_skipped: int = 0
    tiles_full: int = 0
    tiles_partial: int = 0
    reads: int = 0          # elements read from "slow memory" (Q, K, V tiles)
    writes: int = 0         # elements written to "slow memory" (O, lse)
    flops: int = 0          # matmul flops actually executed
    peak_tile_elems: int = 0

    @property
    def computed(self) -> int:
        return self.tiles_full + self.tiles_partial

    @property
    def skipped_fraction(self) -> float:
        return self.tiles_skipped / self.tiles_total if self.tiles_total else 0.0

    @property
    def traffic(self) -> int:
        return self.reads + self.writes


def _online_update(m, l, acc, s, vt):
    """Fold one tile of scores ``s`` (rows x cols, -inf = masked) and values ``vt`` into the
    running (m, l, acc). Rows that have seen only masked scores stay at m = -inf, l = 0."""
    m_new = np.maximum(m, s.max(axis=1))
    m_safe = np.where(np.isfinite(m_new), m_new, 0.0)
    alpha = np.exp(np.where(np.isfinite(m), m - m_safe, -np.inf))
    p = np.exp(s - m_safe[:, None])
    l = alpha * l + p.sum(axis=1)
    acc = alpha[:, None] * acc + p @ vt
    return m_new, l, acc


def attend_partial(q, k, v, *, causal=False, window=None, alibi_slopes=None, q_offset=0,
                   k_start=0, block_q=64, block_k=64, scale=None, dtype=np.float64, stats=None):
    """Online-softmax attention of ``q`` against a *slice* of the keys.

    ``k``/``v`` hold the keys whose absolute positions are ``k_start .. k_start+len-1``. Returns
    the un-normalised partial state ``(m, l, acc)`` - row max, denominator, weighted value sum -
    which :func:`merge_partials` combines exactly across slices, and :func:`finalize` turns into
    the output. Shapes: m, l (H, nq); acc (H, nq, d).
    """
    q, k, v = (np.asarray(x, dtype=dtype) for x in (q, k, v))
    H, nq, d = q.shape
    nk = k.shape[1]
    g = _group(q, k)
    scale = dtype(d ** -0.5 if scale is None else scale)
    stats = stats if stats is not None else Stats()
    m = np.full((H, nq), -np.inf, dtype=dtype)
    l = np.zeros((H, nq), dtype=dtype)
    acc = np.zeros((H, nq, d), dtype=dtype)
    for h in range(H):
        kh, vh = k[h // g], v[h // g]
        slope = None if alibi_slopes is None else alibi_slopes[h]
        for i0 in range(0, nq, block_q):
            i1 = min(i0 + block_q, nq)
            qt = q[h, i0:i1]
            mi = np.full(i1 - i0, -np.inf, dtype=dtype)
            li = np.zeros(i1 - i0, dtype=dtype)
            ai = np.zeros((i1 - i0, d), dtype=dtype)
            stats.reads += (i1 - i0) * d
            for j0 in range(0, nk, block_k):
                j1 = min(j0 + block_k, nk)
                stats.tiles_total += 1
                status = tile_status(i0, i1, k_start + j0, k_start + j1, q_offset, causal, window)
                if status == SKIP:
                    stats.tiles_skipped += 1
                    continue
                stats.reads += 2 * (j1 - j0) * d
                stats.flops += 4 * (i1 - i0) * (j1 - j0) * d
                stats.peak_tile_elems = max(stats.peak_tile_elems, (i1 - i0) * (j1 - j0))
                s = scale * (qt @ kh[j0:j1].T)
                if slope is not None:
                    s = s + alibi_bias(slope, i1 - i0, j1 - j0, q_offset + i0, k_start + j0).astype(dtype)
                if status == PARTIAL:
                    keep = allowed_mask(i1 - i0, j1 - j0, q_offset + i0, causal, window, k_start + j0)
                    s = np.where(keep, s, dtype(-np.inf))
                    stats.tiles_partial += 1
                else:
                    stats.tiles_full += 1
                mi, li, ai = _online_update(mi, li, ai, s, vh[j0:j1])
            m[h, i0:i1], l[h, i0:i1], acc[h, i0:i1] = mi, li, ai
    return m, l, acc


def merge_partials(parts):
    """Exactly combine partial states computed over disjoint key slices (any order, any grouping)."""
    ms = np.stack([p[0] for p in parts])
    m = ms.max(axis=0)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    w = np.exp(np.where(np.isfinite(ms), ms - m_safe, -np.inf))
    l = sum(w[i] * parts[i][1] for i in range(len(parts)))
    acc = sum(w[i][..., None] * parts[i][2] for i in range(len(parts)))
    return m, l, acc


def finalize(m, l, acc):
    """Normalise a partial state: (out, lse). Rows with no visible key give 0 and -inf."""
    ok = l > 0
    out = np.where(ok[..., None], acc / np.where(ok, l, 1.0)[..., None], 0.0)
    lse = np.where(ok, np.where(np.isfinite(m), m, 0.0) + np.log(np.where(ok, l, 1.0)), -np.inf)
    return out, lse


def flash_forward(q, k, v, *, causal=False, window=None, alibi_slopes=None, q_offset=None,
                  block_q=64, block_k=64, scale=None, dtype=np.float64):
    """Tiled attention. Returns (out, lse, stats); ``out`` has the working ``dtype``."""
    q = np.asarray(q)
    nq, nk = q.shape[1], np.asarray(k).shape[1]
    off = resolve_offset(q_offset, nq, nk)
    stats = Stats()
    m, l, acc = attend_partial(q, k, v, causal=causal, window=window, alibi_slopes=alibi_slopes,
                               q_offset=off, block_q=block_q, block_k=block_k, scale=scale,
                               dtype=dtype, stats=stats)
    out, lse = finalize(m, l, acc)
    stats.writes += out.size + lse.size
    return out.astype(dtype), lse.astype(dtype), stats


def flash_backward(q, k, v, out, lse, dout, *, causal=False, window=None, alibi_slopes=None,
                   q_offset=None, block_q=64, block_k=64, scale=None, dtype=np.float64):
    """FlashAttention-2 backward: needs only (q, k, v, out, lse, dout); recomputes P per tile.

    Returns (dq, dk, dv). Only O(block_q * block_k) scores live at a time.
    """
    q, k, v, out, lse, dout = (np.asarray(x, dtype=dtype) for x in (q, k, v, out, lse, dout))
    H, nq, d = q.shape
    nk = k.shape[1]
    g = _group(q, k)
    off = resolve_offset(q_offset, nq, nk)
    scale = dtype(d ** -0.5 if scale is None else scale)
    D = (dout * out).sum(axis=-1)                        # (H, nq): rowsum(dO * O)
    lse_safe = np.where(np.isfinite(lse), lse, np.inf)   # fully-masked rows -> P = 0
    dq, dk, dv = np.zeros_like(q), np.zeros_like(k), np.zeros_like(v)
    for h in range(H):
        kv = h // g
        slope = None if alibi_slopes is None else alibi_slopes[h]
        for j0 in range(0, nk, block_k):
            j1 = min(j0 + block_k, nk)
            kt, vt = k[kv, j0:j1], v[kv, j0:j1]
            dkt = np.zeros_like(kt)
            dvt = np.zeros_like(vt)
            for i0 in range(0, nq, block_q):
                i1 = min(i0 + block_q, nq)
                status = tile_status(i0, i1, j0, j1, off, causal, window)
                if status == SKIP:
                    continue
                qt, dot = q[h, i0:i1], dout[h, i0:i1]
                s = scale * (qt @ kt.T)
                if slope is not None:
                    s = s + alibi_bias(slope, i1 - i0, j1 - j0, off + i0, j0).astype(dtype)
                if status == PARTIAL:
                    keep = allowed_mask(i1 - i0, j1 - j0, off + i0, causal, window, j0)
                    s = np.where(keep, s, dtype(-np.inf))
                p = np.exp(s - lse_safe[h, i0:i1, None])
                dvt += p.T @ dot
                dp = dot @ vt.T
                ds = p * (dp - D[h, i0:i1, None])
                dq[h, i0:i1] += scale * (ds @ kt)
                dkt += scale * (ds.T @ qt)
            dk[kv, j0:j1] += dkt
            dv[kv, j0:j1] += dvt
    return dq, dk, dv
