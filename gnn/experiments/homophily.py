"""Accuracy of MLP / GCN / GraphSAGE / GAT as edge homophily goes from 0.1 to 0.9.

Features carry the same (modest) signal in every graph, so the MLP column is flat up to
sampling noise; every difference between columns comes from the edges.

    PYTHONPATH=src python experiments/homophily.py [--seeds 6]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from gnn import GNN, build_ops, edge_homophily, fit, make_split, sbm_graph  # noqa: E402

KINDS = ("mlp", "gcn", "sage", "gat")
HS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def run(seeds: int):
    table = {k: {h: [] for h in HS} for k in KINDS}
    measured = {h: [] for h in HS}
    for h in HS:
        for s in range(seeds):
            g = sbm_graph(n=400, k=4, avg_degree=10, homophily=h, feat_dim=16, seed=100 + s)
            ops = build_ops(g.adj)
            tr, va, te = make_split(g.y, per_class=20, n_val=100, seed=s)
            measured[h].append(edge_homophily(g.adj, g.y))
            for kind in KINDS:
                m = GNN(kind, 16, 32, 4, depth=2, seed=s)
                table[kind][h].append(fit(m, g.x, g.y, ops, tr, va, te).test_acc)
        print(f"h={h} done", flush=True)
    return table, measured


def render(table, measured, seeds):
    lines = [f"test accuracy, mean +- std over {seeds} graphs (400 nodes, 4 classes, 20 labels/class)",
             "", "edge homophily (target / measured) | " + " | ".join(f"{k:>11}" for k in KINDS),
             "-" * 80]
    for h in HS:
        cells = [f"{np.mean(table[k][h]):.3f}+-{np.std(table[k][h]):.3f}" for k in KINDS]
        lines.append(f"{h:.1f} / {np.mean(measured[h]):.2f}".ljust(35) + "| " + " | ".join(cells))
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    a = ap.parse_args()
    table, measured = run(a.seeds)
    out = render(table, measured, a.seeds)
    print(out)
    here = os.path.dirname(__file__)
    open(os.path.join(here, "homophily_results.txt"), "w").write(out + "\n")
    json.dump({k: {str(h): v for h, v in d.items()} for k, d in table.items()},
              open(os.path.join(here, "homophily_results.json"), "w"), indent=1)
