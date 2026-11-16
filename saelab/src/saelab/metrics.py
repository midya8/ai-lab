"""Scoring a sparse autoencoder against known ground-truth features."""

from __future__ import annotations

import numpy as np


def _unit(w: np.ndarray) -> np.ndarray:
    return w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)


def cosine_matrix(true_dirs: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """(n_true, n_latents) cosine similarity between true features and decoder rows."""
    return _unit(true_dirs) @ _unit(dec).T


def feature_recovery(true_dirs: np.ndarray, dec: np.ndarray, thresh: float = 0.9) -> dict:
    """How many true features have a close decoder row?

    ``best``: per-feature best cosine. ``recovered``: fraction with best >= thresh.
    ``matched``: fraction recovered by a *distinct* latent under greedy one-to-one matching
    (a single latent cannot claim two features).
    """
    c = cosine_matrix(true_dirs, dec)
    best = c.max(axis=1)
    cc = c.copy()
    matched = 0
    for _ in range(min(cc.shape)):
        i, j = np.unravel_index(np.argmax(cc), cc.shape)
        if cc[i, j] < thresh:
            break
        matched += 1
        cc[i, :] = -2.0
        cc[:, j] = -2.0
    n = true_dirs.shape[0]
    return {"mean_best_cos": float(best.mean()), "recovered": float((best >= thresh).mean()),
            "matched": matched / n}


def latent_purity(true_dirs: np.ndarray, dec: np.ndarray, thresh: float = 0.9,
                  alive: np.ndarray | None = None) -> float:
    """Fraction of (alive) latents whose decoder row is close to some true feature."""
    c = cosine_matrix(true_dirs, dec)
    ok = c.max(axis=0) >= thresh
    if alive is not None:
        if not alive.any():
            return 0.0
        ok = ok[alive]
    return float(ok.mean())


def fvu(x: np.ndarray, xh: np.ndarray) -> float:
    """Fraction of variance unexplained (0 = perfect, 1 = no better than predicting the mean)."""
    var = ((x - x.mean(axis=0)) ** 2).sum()
    return float(((x - xh) ** 2).sum() / max(var, 1e-12))


def l0(a: np.ndarray) -> float:
    """Average number of non-zero latents per sample."""
    return float((a > 0).sum(axis=1).mean())


def dead_latents(a: np.ndarray) -> np.ndarray:
    """Boolean mask of latents that never fired on this batch."""
    return ~(a > 0).any(axis=0)


def coefficient_ratio(true_a: np.ndarray, learned_a: np.ndarray, true_dirs: np.ndarray,
                      dec: np.ndarray, thresh: float = 0.9) -> float:
    """Mean (learned activation / true coefficient) over recovered features, when both fire.

    1.0 means magnitudes are read out faithfully; < 1 is *shrinkage* (the classic L1 side effect).
    Samples where the true feature fires but its latent stays silent are excluded here; they are
    counted by ``feature_recall``. Returns nan when nothing was recovered.
    """
    c = cosine_matrix(true_dirs, dec)
    j = c.argmax(axis=1)
    good = c.max(axis=1) >= thresh
    ratios = []
    for i in np.where(good)[0]:
        on = (true_a[:, i] > 0) & (learned_a[:, j[i]] > 0)
        if on.any():
            ratios.append(float(np.mean(learned_a[on, j[i]] / true_a[on, i])))
    return float(np.mean(ratios)) if ratios else float("nan")


def feature_recall(true_a: np.ndarray, learned_a: np.ndarray, true_dirs: np.ndarray,
                   dec: np.ndarray, thresh: float = 0.9) -> float:
    """Of the (sample, feature) events where a recovered feature truly fires, the fraction whose
    matched latent also fires. Fixed-k TopK loses events on samples that have more than k features."""
    c = cosine_matrix(true_dirs, dec)
    j = c.argmax(axis=1)
    good = np.where(c.max(axis=1) >= thresh)[0]
    hit = tot = 0
    for i in good:
        on = true_a[:, i] > 0
        tot += int(on.sum())
        hit += int((learned_a[on, j[i]] > 0).sum())
    return hit / tot if tot else float("nan")
