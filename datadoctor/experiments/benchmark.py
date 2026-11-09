"""Benchmark: how well do label-error detectors recover injected noise?

    PYTHONPATH=src python experiments/benchmark.py            # full (3 seeds)
    PYTHONPATH=src python experiments/benchmark.py --quick    # 1 seed

Everything is synthetic (known ground truth) and uses few seeds, so differences of a
couple of hundredths are noise. Writes experiments/results.md.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from datadoctor.confident import find_label_issues
from datadoctor.evaluate import cleaning_effect, compare_detectors
from datadoctor.models import KNNClassifier, SoftmaxRegression
from datadoctor.synth import inject_label_noise, make_blobs, make_modes

DATASETS = {
    "blobs (overlap)": lambda s: make_blobs(n=2400, seed=s),
    "two-mode classes": lambda s: make_modes(n=2400, sep=1.5, seed=s),
}
MODELS = {"softmax": SoftmaxRegression, "kNN": KNNClassifier}
NOISE = [("symmetric", 0.10), ("symmetric", 0.20), ("symmetric", 0.40), ("pairwise", 0.30), ("asymmetric", 0.30)]
KEYS = ("auroc", "p_at_k", "f1_both", "f1_rate", "est", "noisy", "removed", "oracle", "base_auroc", "knn_auroc")


def run(seeds):
    rows = []
    for dname, mk in DATASETS.items():
        for mname, factory in MODELS.items():
            for kind, rate in NOISE:
                acc = {k: [] for k in KEYS}
                for s in seeds:
                    X, y = mk(s)
                    n_tr = 1800
                    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
                    yn, fl = inject_label_noise(ytr, rate, kind=kind, seed=s)
                    r = compare_detectors(Xtr, yn, fl, seed=s, model_factory=factory)
                    acc["auroc"].append(r["ranking"]["CV margin"]["auroc"])
                    acc["p_at_k"].append(r["ranking"]["CV margin"]["p_at_k"])
                    acc["base_auroc"].append(r["ranking"]["in-sample loss"]["auroc"])
                    acc["knn_auroc"].append(r["ranking"]["kNN disagreement"]["auroc"])
                    acc["f1_both"].append(r["masks"]["both"]["f1"])
                    acc["f1_rate"].append(r["masks"]["prune_by_noise_rate"]["f1"])
                    acc["est"].append(r["estimated_noise_rate"])
                    flagged = find_label_issues(r["probs"], yn, "prune_by_noise_rate")
                    c = cleaning_effect(Xtr, yn, ytr, flagged, Xte, yte, seed=s)
                    acc["noisy"].append(c["noisy"])
                    acc["removed"].append(c["removed_flagged"])
                    acc["oracle"].append(c["oracle_clean"])
                row = (dname, mname, kind, rate, {k: float(np.mean(v)) for k, v in acc.items()})
                rows.append(row)
                print(dname, mname, kind, rate, {k: round(v, 3) for k, v in row[4].items()}, flush=True)
    return rows


def render(rows, nseeds):
    L = [
        f"Mean over {nseeds} seed(s), 1800 training rows, 600 clean test rows. AUROC / P@k are for the CV-margin "
        "ranking from the named model; the two baselines are model-independent (in-sample softmax loss, k-NN label disagreement).",
        "",
        "| data | model | noise | true -> est. rate | AUROC margin | AUROC in-sample | AUROC kNN-disagree | P@k | F1 both | F1 noise-rate | test acc: noisy -> pruned (oracle) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, m, kind, rate, a in rows:
        L.append(
            f"| {d} | {m} | {kind} {rate:.0%} | {rate:.2f} -> {a['est']:.2f} | {a['auroc']:.3f} | {a['base_auroc']:.3f} | "
            f"{a['knn_auroc']:.3f} | {a['p_at_k']:.3f} | {a['f1_both']:.3f} | {a['f1_rate']:.3f} | "
            f"{a['noisy']:.3f} -> {a['removed']:.3f} ({a['oracle']:.3f}) |"
        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = [0] if a.quick else [0, 1, 2]
    rows = run(seeds)
    md = render(rows, len(seeds))
    out = os.path.join(os.path.dirname(__file__), "results_quick.md" if a.quick else "results.md")
    open(out, "w").write(md)
    print("\n" + md)
