"""Recall/latency/memory on two synthetic datasets + a filtered-search experiment -> results.md.

Setup is fixed before looking at results: n=20000, dim=64, 200 queries (data points + 10% noise),
k=10, ground truth from the exact FlatIndex. HNSW M=16, efConstruction=100; IVF nlist=sqrt(n)=141.
Single-threaded Python loops over single queries, so latencies include Python overhead and are
only meaningful relative to each other, on this machine, on this run.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vecdb import FlatIndex, HNSWIndex, recall_at_k  # noqa: E402
from vecdb.bench import make_data, run_bench, time_queries, to_markdown  # noqa: E402

N, DIM, NQ, K = 20000, 64, 200, 10
OUT = os.path.join(os.path.dirname(__file__), "results.md")


def filtered_experiment():
    """Graph walk vs exact scan of the matching rows, across filter selectivities."""
    X, lab, Q = make_data("clusters", N, DIM, NQ, seed=0)
    h = HNSWIndex(DIM, M=16, ef_construction=100, seed=0)
    h.add(X)
    flat = FlatIndex(DIM)
    flat.add(X)
    rng = np.random.default_rng(7)
    cases = []
    for frac in (0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.005):
        cases.append((f"random {frac:.1%}", rng.random(N) < frac))
    cl = np.unique(lab)
    cases.append(("1 cluster", lab == cl[0]))
    cases.append(("5 clusters", np.isin(lab, cl[:5])))
    cases.append(("25 clusters", np.isin(lab, cl[:25])))
    rows = []
    for name, mask in cases:
        gt, _ = flat.search(Q, K, mask=mask)
        res = {}
        for label, bf in (("walk", 0.0), ("exact", 1.01)):
            h.brute_force_below = bf
            ms, ids = time_queries(h, Q, K, ef=64, mask=mask)
            res[label] = (recall_at_k(ids, gt), ms, h.last_stats["scanned"])
        rows.append((name, mask.mean(), res))
    ms_flat, _ = time_queries(flat, Q, K)
    return rows, ms_flat


def filtered_markdown(rows, ms_flat):
    out = [f"## Filtered search on HNSW (clusters, ef=64; unfiltered exact scan = {ms_flat:.2f} ms/query)\n",
           "`walk` traverses the graph and collects only matching nodes; `exact` scans just the matching rows.\n",
           "| filter | matching rows | walk recall | walk ms | walk evals | exact ms | exact evals | faster |",
           "|---|---|---|---|---|---|---|---|"]
    for name, sel, r in rows:
        w, e = r["walk"], r["exact"]
        out.append(f"| {name} | {sel:.2%} | {w[0]:.3f} | {w[1]:.2f} | {w[2]:.0f} | {e[1]:.2f} | {e[2]:.0f} | "
                   f"{'walk' if w[1] < e[1] else 'exact'} |")
    return "\n".join(out) + "\n"


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "filtered":      # refresh only the filtered section
        text = open(OUT).read()
        head = text[:text.index("## Filtered search")]
        open(OUT, "w").write(head + filtered_markdown(*filtered_experiment()))
        print("rewrote filtered section")
        return
    parts = ["# vecdb benchmark results\n",
             "Synthetic data, k=10, 200 queries, ground truth = exact search. Single-process Python; "
             "latency is per single-query call and includes interpreter overhead. Bytes/vector counts the "
             "index structures (ids, centroids, codebooks, graph links at 4 bytes/link), not a Python process.\n"]
    for kind in ("clusters", "gaussian"):
        t = time.time()
        rows = run_bench(kind, n=N, dim=DIM, nq=NQ, k=K, log=lambda s: print(kind, s, flush=True))
        parts.append(f"\n## {kind} (n={N}, dim={DIM})\n")
        parts.append(to_markdown(rows) + "\n")
        print(kind, "total", round(time.time() - t), "s", flush=True)
    parts.append("\n" + filtered_markdown(*filtered_experiment()))
    with open(OUT, "w") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
