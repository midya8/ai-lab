"""Split-KV decoding ("flash-decoding"): one query token against a long cache.

With a single query row there is no parallelism across query tiles, so the key axis is cut into
``num_splits`` slices that are processed independently and merged with the exact log-sum-exp
combination in :func:`flashlab.flash.merge_partials`. The result is identical (to rounding) to
attending over the whole cache at once.
"""
from __future__ import annotations

import numpy as np

from .flash import Stats, attend_partial, finalize, merge_partials
from .reference import resolve_offset


def split_ranges(nk: int, num_splits: int):
    """Contiguous, near-equal, non-empty-when-possible key ranges covering [0, nk)."""
    num_splits = max(1, min(num_splits, nk)) if nk else 1
    edges = np.linspace(0, nk, num_splits + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def flash_decode(q, k, v, *, num_splits=4, causal=True, window=None, alibi_slopes=None,
                 q_offset=None, block_k=64, scale=None, dtype=np.float64):
    """Attention of ``q`` (H, nq, d) over a cache ``k, v`` (Hkv, nk, d) computed as independent
    key splits then merged. Returns (out, lse, per-split Stats list)."""
    k, v = np.asarray(k), np.asarray(v)
    nq, nk = np.asarray(q).shape[1], k.shape[1]
    off = resolve_offset(q_offset, nq, nk)
    parts, stats = [], []
    for a, b in split_ranges(nk, num_splits):
        st = Stats()
        parts.append(attend_partial(q, k[:, a:b], v[:, a:b], causal=causal, window=window,
                                    alibi_slopes=alibi_slopes, q_offset=off, k_start=a,
                                    block_q=max(nq, 1), block_k=block_k, scale=scale,
                                    dtype=dtype, stats=st))
        stats.append(st)
    out, lse = finalize(*merge_partials(parts))
    return out, lse, stats
