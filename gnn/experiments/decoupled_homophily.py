"""Decoupling study, part 2: APPNP / SGC / label propagation vs MLP / GCN across homophily.

Same graphs, splits and seeds as homophily.py (SBM, seeds 100+s). Label propagation uses the
edges and the 80 training labels only; every other model also sees the features.

    PYTHONPATH=src python experiments/decoupled_homophily.py [--seeds 6]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gnn import (APPNP, GNN, SGC, build_ops, fit, label_propagation, make_split,  # noqa: E402
                    sbm_graph)

HS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
NAMES = ("mlp", "gcn", "sgc k=2", "appnp k=10", "label-prop")


def run(seeds):
    t = {n: {h: [] for h in HS} for n in NAMES}
    for h in HS:
        for s in range(seeds):
            g = sbm_graph(n=400, k=4, avg_degree=10, homophily=h, feat_dim=16, seed=100 + s)
            ops = build_ops(g.adj)
            tr, va, te = make_split(g.y, per_class=20, n_val=100, seed=s)
            def go(m):
                return fit(m, g.x, g.y, ops, tr, va, te).test_acc
            t["mlp"][h].append(go(GNN("mlp", 16, 32, 4, depth=2, seed=s)))
            t["gcn"][h].append(go(GNN("gcn", 16, 32, 4, depth=2, seed=s)))
            t["sgc k=2"][h].append(go(SGC(16, 4, k=2, seed=s)))
            t["appnp k=10"][h].append(go(APPNP(16, 32, 4, k=10, alpha=0.1, seed=s)))
            f = label_propagation(ops, g.y, tr, g.k)
            t["label-prop"][h].append(float((f[te].argmax(1) == g.y[te]).mean()))
        print(f"h={h} done", flush=True)
    return t


def render(t, seeds):
    lines = [f"test accuracy, mean +- std over {seeds} graphs (400 nodes, 4 classes, 20 labels/class)", "",
             "edge homophily | " + " | ".join(f"{n:>13}" for n in NAMES), "-" * 85]
    for h in HS:
        lines.append(f"{h:.1f}".ljust(15) + "| " + " | ".join(
            f"{np.mean(t[n][h]):.3f}+-{np.std(t[n][h]):.3f}" for n in NAMES))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    out = render(run(a.seeds), a.seeds)
    print(out)
    open(os.path.join(os.path.dirname(__file__), "decoupled_homophily_results.txt"), "w").write(out + "\n")
