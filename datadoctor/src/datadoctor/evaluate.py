"""Score label-error detectors against injected ground truth, and measure what cleaning buys."""

from __future__ import annotations

import numpy as np

from .confident import find_label_issues, label_quality_scores
from .cv import cross_val_probs
from .models import SoftmaxRegression


def precision_at_k(suspicion: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    k = int(truth.sum()) if k is None else k
    if k <= 0:
        return 0.0
    top = np.argsort(-suspicion, kind="stable")[:k]
    return float(truth[top].mean())


def average_precision(suspicion: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(-suspicion, kind="stable")
    hits = truth[order].astype(float)
    if hits.sum() == 0:
        return 0.0
    prec = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((prec * hits).sum() / hits.sum())


def auroc(suspicion: np.ndarray, truth: np.ndarray) -> float:
    """Probability a random true error outranks a random clean label (ties count half)."""
    pos = suspicion[truth]
    neg = suspicion[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def mask_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int((pred & truth).sum())
    p = tp / pred.sum() if pred.sum() else 0.0
    r = tp / truth.sum() if truth.sum() else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"flagged": int(pred.sum()), "precision": p, "recall": r, "f1": f}


def knn_disagreement(X: np.ndarray, y: np.ndarray, k: int = 10) -> np.ndarray:
    """Fraction of the k nearest neighbours (leave-one-out) whose label differs. A common baseline."""
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-12, 1.0, X.std(0))
    sq = (Z * Z).sum(1)
    out = np.empty(len(y))
    for s in range(0, len(Z), 512):
        D = np.maximum(sq[s : s + 512, None] + sq[None, :] - 2 * Z[s : s + 512] @ Z.T, 0)
        D[np.arange(D.shape[0]), np.arange(s, s + D.shape[0])] = np.inf
        nn = np.argpartition(D, k - 1, axis=1)[:, :k]
        out[s : s + 512] = (y[nn] != y[s : s + 512, None]).mean(axis=1)
    return out


def in_sample_loss(X: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    """Train on everything, rank by loss on the training labels (memorisation hides errors)."""
    m = SoftmaxRegression().fit(X, y, n_classes=K)
    P = m.predict_proba(X)
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def compare_detectors(X, y_noisy, flipped, k_folds: int = 5, repeats: int = 1, seed: int = 0, model_factory=None) -> dict:
    """Ranking metrics (P@k, AP, AUROC) for every detector, plus mask metrics for pruning rules."""
    K = int(y_noisy.max()) + 1
    P = cross_val_probs(X, y_noisy, model_factory, k=k_folds, seed=seed, repeats=repeats, n_classes=K)
    rng = np.random.default_rng(seed)
    rankers = {
        "random": rng.random(len(y_noisy)),
        "in-sample loss": in_sample_loss(X, y_noisy, K),
        "kNN disagreement": knn_disagreement(X, y_noisy),
        "CV self-confidence": -label_quality_scores(P, y_noisy, "self_confidence"),
        "CV margin": -label_quality_scores(P, y_noisy, "margin"),
        "CV entropy-weighted": -label_quality_scores(P, y_noisy, "entropy"),
    }
    out = {"ranking": {}, "masks": {}}
    for name, s in rankers.items():
        out["ranking"][name] = {
            "p_at_k": precision_at_k(s, flipped),
            "ap": average_precision(s, flipped),
            "auroc": auroc(s, flipped),
        }
    for method in ("argmax", "confident_joint", "prune_by_class", "prune_by_noise_rate", "both"):
        out["masks"][method] = mask_scores(find_label_issues(P, y_noisy, method), flipped)
    from .confident import calibrate_joint, confident_joint

    Cc = calibrate_joint(confident_joint(P, y_noisy)[0], y_noisy)
    out["estimated_noise_rate"] = float(1.0 - np.trace(Cc) / Cc.sum())
    out["true_noise_rate"] = float(flipped.mean())
    out["probs"] = P
    return out


def cleaning_effect(X, y_noisy, y_true, flagged: np.ndarray, X_test, y_test, seed: int = 0) -> dict:
    """Test accuracy of a model trained on: noisy labels / flagged rows removed / oracle-clean data."""
    K = int(y_true.max()) + 1

    def acc(Xtr, ytr):
        m = SoftmaxRegression(seed=seed).fit(Xtr, ytr, n_classes=K)
        return float((m.predict(X_test) == y_test).mean())

    keep = ~flagged
    return {
        "noisy": acc(X, y_noisy),
        "removed_flagged": acc(X[keep], y_noisy[keep]),
        "oracle_clean": acc(X, y_true),
    }
