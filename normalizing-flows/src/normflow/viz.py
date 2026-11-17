"""Dependency-free ASCII scatter/density plots (so README figures regenerate without matplotlib)."""
from __future__ import annotations

import numpy as np

_RAMP = " .:-=+*#%@"


def ascii_density(points: np.ndarray, lim: float = 3.0, width: int = 48, height: int = 22) -> str:
    """2-D histogram of ``points`` over [-lim, lim]^2 rendered with a 10-level character ramp
    (log-scaled so faint bridges between modes stay visible).  y grows upward."""
    pts = np.asarray(points)
    h, _, _ = np.histogram2d(pts[:, 1], pts[:, 0], bins=[height, width], range=[[-lim, lim], [-lim, lim]])
    if h.max() == 0:
        return "\n".join(" " * width for _ in range(height))
    v = np.log1p(h) / np.log1p(h.max())
    idx = np.where(h > 0, np.clip((v * (len(_RAMP) - 1)).round().astype(int), 1, len(_RAMP) - 1), 0)
    return "\n".join("".join(_RAMP[i] for i in row) for row in idx[::-1])
