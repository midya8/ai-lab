"""Confident learning (Northcutt et al., 2021), implemented from the paper's definitions.

Given out-of-sample probabilities ``P`` (n x K) and the given, possibly noisy labels
``y``, estimate which labels are wrong without knowing the truth:

1. *Per-class threshold* t_j = mean of P[i, j] over examples labelled j (how confident
   the model usually is when a j really is a j).
2. *Confident joint* C[i][j] counts examples labelled i whose probability for class j
   clears t_j (taking the most probable such j). Off-diagonal mass = probable errors.
3. *Calibration* rescales rows so they sum to the label counts, then the total to n.
4. The calibrated joint gives the noise matrix P(label | true class) and the class prior.
5. *Pruning* picks how many examples to flag per (given, true) pair and which ones.
"""

from __future__ import annotations

import numpy as np


def class_thresholds(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """t_j = average self-confidence of class j. A class with no examples gets 1.0."""
    K = P.shape[1]
    t = np.ones(K)
    for j in range(K):
        sel = y == j
        if sel.any():
            t[j] = P[sel, j].mean()
    return t


def confident_joint(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (C, assigned): raw integer counts and each example's inferred true class.

    ``assigned[i] == -1`` when no class clears its threshold (the example is not counted).
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    n, K = P.shape
    t = class_thresholds(P, y)
    ok = P >= t[None, :] - 1e-12
    masked = np.where(ok, P, -np.inf)
    best = masked.argmax(axis=1)
    has = ok.any(axis=1)
    assigned = np.where(has, best, -1)
    C = np.zeros((K, K), dtype=int)
    for i in np.flatnonzero(has):
        C[y[i], assigned[i]] += 1
    return C, assigned


def calibrate_joint(C: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scale rows of ``C`` to the observed label counts, then the whole matrix to sum to n."""
    C = np.asarray(C, dtype=float)
    K = C.shape[0]
    counts = np.bincount(y, minlength=K).astype(float)
    rows = C.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    Cc = C / rows * counts[:, None]
    total = Cc.sum()
    return Cc / total * len(y) if total > 0 else Cc


def estimate_noise_matrix(C_cal: np.ndarray) -> np.ndarray:
    """Noise matrix N[i][j] = P(given label i | true class j); columns sum to 1."""
    cols = C_cal.sum(axis=0, keepdims=True)
    cols[cols == 0] = 1.0
    return C_cal / cols


def estimate_prior(C_cal: np.ndarray) -> np.ndarray:
    """Estimated *true* class distribution (column sums of the calibrated joint)."""
    s = C_cal.sum(axis=0)
    return s / s.sum()


def self_confidence(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return P[np.arange(len(y)), y]


def normalized_margin(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """P[given] - max P[other]. Negative means the model prefers another class."""
    n = len(y)
    own = P[np.arange(n), y]
    Q = P.copy()
    Q[np.arange(n), y] = -np.inf
    return own - Q.max(axis=1)


def label_quality_scores(P: np.ndarray, y: np.ndarray, method: str = "self_confidence") -> np.ndarray:
    """Higher = more likely correct. ``method``: self_confidence | margin | entropy."""
    if method == "self_confidence":
        return self_confidence(P, y)
    if method == "margin":
        return normalized_margin(P, y)
    if method == "entropy":
        # self-confidence discounted by how unsure the model is overall (CL's "confidence weighted entropy")
        Pc = np.clip(P, 1e-12, 1.0)
        H = -(Pc * np.log(Pc)).sum(axis=1)
        Hn = H / np.log(P.shape[1]) if P.shape[1] > 1 else H
        return self_confidence(P, y) / np.maximum(Hn, 1e-3)
    raise ValueError(f"unknown scoring method: {method}")


def rank_label_issues(P: np.ndarray, y: np.ndarray, method: str = "margin") -> np.ndarray:
    """Indices sorted from most to least suspicious label."""
    return np.argsort(label_quality_scores(P, y, method), kind="stable")


def find_label_issues(P: np.ndarray, y: np.ndarray, method: str = "both") -> np.ndarray:
    """Boolean mask of probable label errors.

    method:
      ``argmax``                 given label != the model's top class
      ``confident_joint``        off-diagonal members of the confident joint
      ``prune_by_class``         per given class i, drop the (n_i * noise rate) lowest self-confidence
      ``prune_by_noise_rate``    per (i, j) pair drop the C_cal[i][j] largest P[j] - P[i] margins
      ``both``                   intersection of the two pruning rules (default; most precise)
    """
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=int)
    n, K = P.shape
    if method == "argmax":
        return P.argmax(axis=1) != y
    C, assigned = confident_joint(P, y)
    if method == "confident_joint":
        return (assigned >= 0) & (assigned != y)
    Cc = calibrate_joint(C, y)
    by_class = np.zeros(n, dtype=bool)
    by_rate = np.zeros(n, dtype=bool)
    if method in ("prune_by_class", "both"):
        sc = self_confidence(P, y)
        for i in range(K):
            m = int(round(Cc[i].sum() - Cc[i, i]))
            idx = np.flatnonzero(y == i)
            if m > 0 and len(idx):
                order = idx[np.argsort(sc[idx], kind="stable")]
                by_class[order[:m]] = True
    if method in ("prune_by_noise_rate", "both"):
        for i in range(K):
            idx = np.flatnonzero(y == i)
            for j in range(K):
                if i == j:
                    continue
                m = int(round(Cc[i, j]))
                if m <= 0 or not len(idx):
                    continue
                gap = P[idx, j] - P[idx, i]
                order = idx[np.argsort(-gap, kind="stable")]
                by_rate[order[:m]] = True
    if method == "prune_by_class":
        return by_class
    if method == "prune_by_noise_rate":
        return by_rate
    if method == "both":
        return by_class & by_rate
    raise ValueError(f"unknown method: {method}")
