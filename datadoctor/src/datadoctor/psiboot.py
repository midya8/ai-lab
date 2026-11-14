"""Bootstrap confidence intervals for PSI, so small-sample PSI is not over-read."""
from __future__ import annotations

import numpy as np

from .drift import psi


def psi_ci(a, b, bins: int = 10, n_boot: int = 200, level: float = 0.95, seed: int = 0) -> dict:
    """PSI of b vs a with a percentile bootstrap CI (both samples resampled).

    PSI is biased upward on small samples (empty-ish bins), so the CI lower
    bound is the more honest 'is there drift' number. Percentile bootstrap,
    not bias-corrected.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        raise ValueError("need at least 2 samples per side")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    rng = np.random.default_rng(seed)
    vals = np.array([
        psi(rng.choice(a, a.size), rng.choice(b, b.size), bins=bins) for _ in range(n_boot)
    ])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"psi": psi(a, b, bins=bins), "lo": float(lo), "hi": float(hi), "level": level}


def psi_ci_report(X_train, X_test, names=None, threshold: float = 0.25, **kw) -> list[dict]:
    """Per-feature psi_ci; `confident` means the CI lower bound exceeds threshold."""
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    names = names or [f"x{j}" for j in range(X_train.shape[1])]
    out = []
    for j, n in enumerate(names):
        r = psi_ci(X_train[:, j], X_test[:, j], **kw)
        r.update(feature=n, confident=r["lo"] > threshold)
        out.append(r)
    return sorted(out, key=lambda r: -r["psi"])
