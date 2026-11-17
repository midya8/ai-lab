"""Depth study: does stacking graph layers oversmooth, and what fixes it?

For a GCN of depth L on a homophilous SBM (h=0.8) we train four variants and report test
accuracy plus two diagnostics of the *last hidden layer* (before the classifier):
Dirichlet energy (0 = every feature column is the oversmoothed sqrt-degree vector) and mean
pairwise cosine similarity between nodes (1 = all nodes identical in direction).

    PYTHONPATH=src python experiments/depth.py [--seeds 4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gnn import (GNN, build_ops, dirichlet_energy, fit, make_split,  # noqa: E402
                    mean_pairwise_cosine, sbm_graph)

DEPTHS = (2, 4, 8, 16, 32)
VARIANTS = {
    "plain": dict(residual=False, pairnorm_scale=None),
    "residual": dict(residual=True, pairnorm_scale=None),
    "pairnorm": dict(residual=False, pairnorm_scale=1.0),
    "res+pairnorm": dict(residual=True, pairnorm_scale=1.0),
}


def run(seeds):
    res = {(v, d): {"acc": [], "energy": [], "cos": [], "train": []}
           for v in VARIANTS for d in DEPTHS}
    for s in range(seeds):
        g = sbm_graph(n=400, k=4, avg_degree=10, homophily=0.8, feat_dim=16, seed=200 + s)
        ops = build_ops(g.adj)
        tr, va, te = make_split(g.y, per_class=20, n_val=100, seed=s)
        for v, kw in VARIANTS.items():
            for d in DEPTHS:
                m = GNN("gcn", 16, 32, 4, depth=d, seed=s, **kw)
                r = fit(m, g.x, g.y, ops, tr, va, te, epochs=200)
                m.forward(g.x, ops)
                hlast = m.hidden_states[-1]
                cell = res[(v, d)]
                cell["acc"].append(r.test_acc)
                cell["train"].append(r.train_acc)
                cell["energy"].append(dirichlet_energy(hlast, ops))
                cell["cos"].append(mean_pairwise_cosine(hlast))
        print(f"seed {s} done", flush=True)
    return res


def render(res, seeds):
    lines = [f"GCN depth study, h=0.8, mean over {seeds} graphs; hidden width 32, 200 epochs",
             "test acc (train acc) | Dirichlet energy of last hidden layer | mean pairwise cosine",
             ""]
    head = "variant".ljust(14) + "".join(f"L={d}".rjust(26) for d in DEPTHS)
    lines.append(head)
    for v in VARIANTS:
        row = v.ljust(14)
        for d in DEPTHS:
            c = res[(v, d)]
            row += f"{np.mean(c['acc']):.2f}({np.mean(c['train']):.2f}) {np.mean(c['energy']):.3f} {np.mean(c['cos']):.2f}".rjust(26)
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    out = render(run(a.seeds), a.seeds)
    print(out)
    open(os.path.join(os.path.dirname(__file__), "depth_results.txt"), "w").write(out + "\n")
