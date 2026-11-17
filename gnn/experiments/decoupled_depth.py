"""Decoupling study, part 1: does separating propagation from transformation remove the
depth collapse seen in `depth.py`?

Same graphs as depth.py (SBM, h=0.8, seeds 200+s). "Depth" means propagation hops K for
APPNP/SGC and layers L for GCN/JKNet. Hidden width 32, 200 epochs, test acc (train acc).

    PYTHONPATH=src python experiments/decoupled_depth.py [--seeds 4]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gnn import (APPNP, GNN, JKNet, SGC, build_ops, fit, make_split, sbm_graph)  # noqa: E402

DEPTHS = (2, 8, 16, 32, 64)
MODELS = {
    "gcn (plain)": lambda d, s: GNN("gcn", 16, 32, 4, depth=d, seed=s),
    "jknet (concat)": lambda d, s: JKNet(16, 32, 4, depth=d, seed=s),
    "sgc": lambda d, s: SGC(16, 4, k=d, seed=s),
    "appnp (a=0.1)": lambda d, s: APPNP(16, 32, 4, k=d, alpha=0.1, seed=s),
    "appnp (a=0)": lambda d, s: APPNP(16, 32, 4, k=d, alpha=0.0, seed=s),
}
SKIP = {("gcn (plain)", 64), ("jknet (concat)", 64)}   # layer-stack cost; K=64 hops are cheap


def run(seeds):
    res = {(m, d): {"acc": [], "train": []} for m in MODELS for d in DEPTHS if (m, d) not in SKIP}
    for s in range(seeds):
        g = sbm_graph(n=400, k=4, avg_degree=10, homophily=0.8, feat_dim=16, seed=200 + s)
        ops = build_ops(g.adj)
        tr, va, te = make_split(g.y, per_class=20, n_val=100, seed=s)
        for name, make in MODELS.items():
            for d in DEPTHS:
                if (name, d) in SKIP:
                    continue
                r = fit(make(d, s), g.x, g.y, ops, tr, va, te, epochs=200)
                res[(name, d)]["acc"].append(r.test_acc)
                res[(name, d)]["train"].append(r.train_acc)
        print(f"seed {s} done", flush=True)
    return res


def render(res, seeds):
    lines = [f"depth study with decoupled models, h=0.8, {seeds} graphs, width 32, 200 epochs",
             "cell: test acc (train acc), mean over graphs; '-' = not run", "",
             "model".ljust(16) + "".join(f"{'K/L=' + str(d):>16}" for d in DEPTHS)]
    for m in MODELS:
        row = m.ljust(16)
        for d in DEPTHS:
            c = res.get((m, d))
            row += (f"{np.mean(c['acc']):.2f} ({np.mean(c['train']):.2f})" if c else "-").rjust(16)
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    out = render(run(a.seeds), a.seeds)
    print(out)
    open(os.path.join(os.path.dirname(__file__), "decoupled_depth_results.txt"), "w").write(out + "\n")
