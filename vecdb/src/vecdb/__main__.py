"""python -m vecdb bench|demo"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import Collection, clustered
from .bench import run_bench, to_markdown


def cmd_bench(a) -> int:
    rows = run_bench(a.dataset, a.n, a.dim, a.queries, a.k, a.seed, log=lambda s: print(s, file=sys.stderr))
    print(to_markdown(rows))
    return 0


def cmd_demo(a) -> int:
    X, lab = clustered(a.n, 32, n_clusters=20, spread=0.8, seed=a.seed)
    c = Collection(32, "cosine", "hnsw", M=12, ef_construction=80)
    c.upsert([f"doc{i}" for i in range(a.n)], X, [{"topic": int(l), "year": 2015 + i % 10} for i, l in enumerate(lab)])
    q = X[7] + 0.05
    print(f"{len(c)} vectors, query is a perturbed doc7 (topic {lab[7]})")
    print("\nunfiltered top 3:")
    for h in c.query(q, k=3, ef=64)[0]:
        print(f"  {h.id:8s} cos-dist {h.distance:.4f}  {h.metadata}")
    where = {"year": {"$gte": 2022}, "topic": {"$ne": int(lab[7])}}
    print(f"\nfiltered {where} top 3 ({c.count(where)} rows match):")
    for h in c.query(q, k=3, where=where, ef=64)[0]:
        print(f"  {h.id:8s} cos-dist {h.distance:.4f}  {h.metadata}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="vecdb")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="recall/latency table for flat, IVF-Flat, IVF-PQ and HNSW")
    b.add_argument("--dataset", choices=["clusters", "gaussian"], default="clusters")
    b.add_argument("--n", type=int, default=5000)
    b.add_argument("--dim", type=int, default=64)
    b.add_argument("--queries", type=int, default=100)
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_bench)
    d = sub.add_parser("demo", help="filtered nearest-neighbour search on a small collection")
    d.add_argument("--n", type=int, default=2000)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_demo)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
