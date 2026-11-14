"""Non-analytic 2-D datasets (no closed-form score): judged only by sample-quality metrics."""
from __future__ import annotations

import numpy as np


def spiral(n: int, rng: np.random.Generator, turns: float = 1.5, noise: float = 0.05, scale: float = 0.6):
    """Two-arm Archimedean spiral, roughly unit-variance. Returns (x, arm_label)."""
    arm = rng.integers(0, 2, n)
    u = np.sqrt(rng.random(n))                    # uniform along arc length-ish
    ang = u * turns * 2 * np.pi + arm * np.pi
    r = scale * (0.3 + 2.2 * u)
    x = np.stack([r * np.cos(ang), r * np.sin(ang)], 1) + noise * rng.standard_normal((n, 2))
    return x, arm
