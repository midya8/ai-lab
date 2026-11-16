"""Attention masks, ALiBi bias and exact tile classification.

Conventions (shared by every kernel in this package):

* ``q`` row ``i`` sits at absolute position ``q_offset + i``; key ``j`` sits at absolute position ``j``.
* causal: key ``j`` is visible iff ``j <= qpos``.
* window ``w``: only the ``w`` most recent keys (``qpos - j < w``) are visible when causal;
  when not causal the window is symmetric (``|qpos - j| < w``).
* ALiBi: a bias of ``-slope * |qpos - j|`` is added to the score of head ``h``.
* A query row with no visible key outputs zeros and has ``lse = -inf``.
"""
from __future__ import annotations

import numpy as np

SKIP, FULL, PARTIAL = "skip", "full", "partial"


def allowed_mask(nq: int, nk: int, q_offset: int, causal: bool, window: int | None,
                 k_start: int = 0) -> np.ndarray:
    """Boolean (nq, nk) matrix: True where the query may attend to the key."""
    qpos = np.arange(nq)[:, None] + q_offset
    kpos = np.arange(nk)[None, :] + k_start
    m = np.ones((nq, nk), dtype=bool)
    if causal:
        m &= kpos <= qpos
    if window is not None:
        m &= ((qpos - kpos) < window) if causal else (np.abs(qpos - kpos) < window)
    return m


def alibi_bias(slope: float, nq: int, nk: int, q_offset: int, k_start: int = 0) -> np.ndarray:
    qpos = np.arange(nq)[:, None] + q_offset
    kpos = np.arange(nk)[None, :] + k_start
    return -slope * np.abs(qpos - kpos).astype(np.float64)


def alibi_slopes(n_heads: int) -> np.ndarray:
    """Geometric ALiBi slopes 2^(-8/n), 2^(-16/n), ... (exact for power-of-two head counts)."""
    start = 2.0 ** (-8.0 / n_heads)
    return np.array([start ** (h + 1) for h in range(n_heads)])


def tile_status(i0: int, i1: int, j0: int, j1: int, q_offset: int, causal: bool,
                window: int | None) -> str:
    """Classify the tile of query rows [i0,i1) x absolute key columns [j0,j1) in O(1).

    SKIP    - every score is masked, the tile is never computed
    FULL    - no score is masked, no mask needs to be applied
    PARTIAL - some masked, some visible
    """
    qlo, qhi = i0 + q_offset, i1 - 1 + q_offset      # inclusive absolute query positions
    klo, khi = j0, j1 - 1                             # inclusive absolute key positions
    if causal:
        if klo > qhi:
            return SKIP
        if window is not None and (qlo - khi) >= window:
            return SKIP
        full = khi <= qlo and (window is None or (qhi - klo) < window)
        return FULL if full else PARTIAL
    if window is None:
        return FULL
    gap = max(klo - qhi, qlo - khi, 0)                # smallest |q - k| in the tile
    if gap >= window:
        return SKIP
    widest = max(qhi - klo, khi - qlo)                # largest |q - k| in the tile
    return FULL if widest < window else PARTIAL
