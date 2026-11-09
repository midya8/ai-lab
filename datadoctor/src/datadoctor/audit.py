"""One-call dataset audit and its Markdown / JSON rendering."""

from __future__ import annotations

import numpy as np

from .confident import (
    calibrate_joint,
    confident_joint,
    estimate_noise_matrix,
    estimate_prior,
    find_label_issues,
    label_quality_scores,
)
from .cv import cross_val_probs
from .dups import conflicting_groups, exact_duplicate_groups, near_duplicate_pairs, near_duplicate_texts
from .outliers import find_outliers


def audit(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    texts: list[str] | None = None,
    folds: int = 5,
    repeats: int = 1,
    seed: int = 0,
    method: str = "both",
    max_issues: int | None = None,
    eps: float | None = None,
) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    n, K = len(y), int(y.max()) + 1
    names = class_names or [str(i) for i in range(K)]
    P = cross_val_probs(X, y, k=folds, seed=seed, repeats=repeats, n_classes=K)
    mask = find_label_issues(P, y, method)
    margin = label_quality_scores(P, y, "margin")
    idx = np.flatnonzero(mask)
    idx = idx[np.argsort(margin[idx], kind="stable")]  # worst first
    if max_issues is not None:
        idx = idx[:max_issues]
    C, _ = confident_joint(P, y)
    Cc = calibrate_joint(C, y)
    noise = estimate_noise_matrix(Cc)
    prior = estimate_prior(Cc)
    est_rate = float(1.0 - np.trace(Cc) / Cc.sum()) if Cc.sum() else 0.0

    groups = exact_duplicate_groups(X)
    conflicts = conflicting_groups(groups, y)
    near = near_duplicate_pairs(X, eps=eps)
    out_mask, out_z = find_outliers(X, y)
    result = {
        "n": n,
        "n_classes": K,
        "class_names": names,
        "class_counts": np.bincount(y, minlength=K).tolist(),
        "method": method,
        "folds": folds,
        "cv_accuracy": float((P.argmax(1) == y).mean()),
        "estimated_noise_rate": est_rate,
        "estimated_true_prior": prior.tolist(),
        "noise_matrix": noise.tolist(),
        "confident_joint": C.tolist(),
        "label_issues": [
            {
                "row": int(i),
                "given": names[y[i]],
                "suggested": names[int(P[i].argmax())],
                "p_given": float(P[i, y[i]]),
                "p_suggested": float(P[i].max()),
                "margin": float(margin[i]),
            }
            for i in idx
        ],
        "n_label_issues": int(mask.sum()),
        "exact_duplicate_groups": [[int(i) for i in g] for g in groups],
        "conflicting_duplicate_groups": [[int(i) for i in g] for g in conflicts],
        "near_duplicate_pairs": [{"a": a, "b": b, "distance": d} for a, b, d in near[:200]],
        "n_near_duplicate_pairs": len(near),
        "outliers": [{"row": int(i), "robust_z": float(out_z[i])} for i in np.argsort(-out_z)[: int(out_mask.sum())]],
    }
    if texts is not None:
        tdup = near_duplicate_texts(list(texts))
        result["near_duplicate_texts"] = [{"a": a, "b": b, "jaccard": j} for a, b, j in tdup[:200]]
    return result


def render_markdown(r: dict, top: int = 15) -> str:
    L = []
    L.append("# Dataset audit")
    L.append("")
    L.append(f"{r['n']} rows, {r['n_classes']} classes, {r['folds']}-fold cross-validated accuracy against the given labels: {r['cv_accuracy']:.3f}.")
    L.append("")
    L.append("## Class balance")
    L.append("")
    L.append("| class | given | estimated true share |")
    L.append("|---|---|---|")
    for name, c, p in zip(r["class_names"], r["class_counts"], r["estimated_true_prior"]):
        L.append(f"| {name} | {c} ({c / r['n']:.1%}) | {p:.1%} |")
    L.append("")
    L.append("## Label issues")
    L.append("")
    L.append(
        f"{r['n_label_issues']} rows ({r['n_label_issues'] / r['n']:.1%}) look mislabelled (rule: `{r['method']}`). "
        f"The confident joint estimates an overall noise rate of {r['estimated_noise_rate']:.1%}. "
        "These are *candidates* for a human to review, not verdicts."
    )
    L.append("")
    if r["label_issues"]:
        L.append("| row | given | suggested | P(given) | P(suggested) |")
        L.append("|---|---|---|---|---|")
        for it in r["label_issues"][:top]:
            L.append(f"| {it['row']} | {it['given']} | {it['suggested']} | {it['p_given']:.2f} | {it['p_suggested']:.2f} |")
        if len(r["label_issues"]) > top:
            L.append(f"| ... | {len(r['label_issues']) - top} more | | | |")
        L.append("")
    L.append("## Duplicates")
    L.append("")
    L.append(f"- exact duplicate groups: {len(r['exact_duplicate_groups'])}")
    L.append(f"- groups with conflicting labels: {len(r['conflicting_duplicate_groups'])}")
    for g in r["conflicting_duplicate_groups"][:top]:
        L.append(f"  - rows {g}")
    L.append(f"- near-duplicate pairs (within 5% of the typical nearest-neighbour distance): {r['n_near_duplicate_pairs']}")
    if "near_duplicate_texts" in r:
        L.append(f"- near-duplicate texts (3-word shingle Jaccard >= 0.8): {len(r['near_duplicate_texts'])}")
    L.append("")
    L.append("## Outliers")
    L.append("")
    if r["outliers"]:
        L.append(f"{len(r['outliers'])} rows are far from their own class (robust z > 3.5): " + ", ".join(str(o["row"]) for o in r["outliers"][:top]))
    else:
        L.append("None above robust z 3.5.")
    L.append("")
    return "\n".join(L)
