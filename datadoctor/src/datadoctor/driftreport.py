"""One-call drift triage that combines the per-feature and joint detectors."""
from __future__ import annotations

import numpy as np

from .drift import feature_drift
from .mmd import mmd_drift
from .wasserstein import wasserstein_drift


def drift_report(X_train, X_test, names=None, n_perm: int = 100, seed: int = 0) -> dict:
    """Run KS/PSI, Wasserstein and MMD and classify the shift.

    verdict: 'none' (nothing fires), 'marginal' (some feature flagged by KS/PSI or
    Wasserstein), 'joint-only' (MMD fires but no per-feature detector does, e.g. a
    changed correlation), or 'both'. Thresholds are those of the component
    detectors; the verdict is a triage label, not a statistical test of its own.
    """
    X_train, X_test = np.asarray(X_train, float), np.asarray(X_test, float)
    fd = feature_drift(X_train, X_test, names)
    wd = wasserstein_drift(X_train, X_test, names)
    md = mmd_drift(X_train, X_test, n_perm=n_perm, seed=seed)
    flagged = sorted(set(fd["flagged"]) | set(wd["flagged"]))
    marginal, joint = bool(flagged), bool(md["drifted"])
    verdict = "both" if marginal and joint else "marginal" if marginal else "joint-only" if joint else "none"
    return {"verdict": verdict, "flagged_features": flagged, "feature_drift": fd,
            "wasserstein": wd, "mmd": md}
