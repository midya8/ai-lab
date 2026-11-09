"""Recall / latency / memory benchmark harness shared by the CLI and experiments/."""
from __future__ import annotations

import time

import numpy as np

from .datasets import clustered, gaussian, queries_like, recall_at_k
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .ivf import IVFIndex


def make_data(kind: str, n: int, dim: int, nq: int, seed: int = 0):
    if kind == "clusters":
        X, lab = clustered(n, dim, n_clusters=max(10, n // 200), spread=1.0, seed=seed)
    elif kind == "gaussian":
        X, lab = gaussian(n, dim, seed=seed), None
    else:
        raise ValueError("dataset must be 'clusters' or 'gaussian'")
    return X, lab, queries_like(X, nq, noise=0.1, seed=seed + 1)


def time_queries(index, Q, k, repeats: int = 2, **params):
    """Single-query latency (ms, best of ``repeats`` passes) plus the results of the last pass."""
    best = float("inf")
    ids = None
    for _ in range(repeats):
        out_i = []
        t = time.perf_counter()
        for q in Q:
            i, _ = index.search(q[None, :], k, **params)
            out_i.append(i[0])
        best = min(best, (time.perf_counter() - t) / len(Q) * 1000)
        ids = np.stack(out_i)
    return best, ids


def run_bench(kind="clusters", n=20000, dim=64, nq=200, k=10, seed=0, hnsw_M=16, hnsw_efc=100,
              nlist=None, pq_m=16, log=print):
    """Rows of dicts: method, setting, recall@k, ms/query, scanned/query, bytes/vector, build_s."""
    X, _, Q = make_data(kind, n, dim, nq, seed)
    nlist = nlist or int(round(np.sqrt(n)))
    rows = []
    flat = FlatIndex(dim)
    t = time.perf_counter(); flat.add(X); build = time.perf_counter() - t
    ms, gt = time_queries(flat, Q, k)
    rows.append(dict(method="flat (exact)", setting="-", recall=1.0, ms=ms, scanned=n,
                     bytes=flat.memory_bytes() / n, build_s=build))
    log(f"flat done {ms:.2f} ms/query")

    ivf = IVFIndex(dim, nlist=nlist, seed=seed)
    ivf.add(X); t = time.perf_counter(); ivf.train(); build = time.perf_counter() - t
    for npb in (1, 2, 4, 8, 16, 32):
        ms, ids = time_queries(ivf, Q, k, nprobe=npb)
        rows.append(dict(method=f"IVF-Flat nlist={nlist}", setting=f"nprobe={npb}", recall=recall_at_k(ids, gt),
                         ms=ms, scanned=ivf.last_stats["scanned"], bytes=ivf.memory_bytes() / n, build_s=build))
    log("ivf-flat done")

    pq = IVFIndex(dim, nlist=nlist, pq_m=pq_m, keep_raw=True, seed=seed)
    pq.add(X); t = time.perf_counter(); pq.train(); build = time.perf_counter() - t
    pq_only_bytes = (pq.memory_bytes() - pq._raw.nbytes) / n
    for npb in (8, 32):
        for refine in (None, 4, 16):
            ms, ids = time_queries(pq, Q, k, nprobe=npb, refine=refine)
            tag = "ADC only" if refine is None else f"ADC + exact re-rank x{refine}"
            rows.append(dict(method=f"IVF-PQ m={pq_m}", setting=f"nprobe={npb}, {tag}", recall=recall_at_k(ids, gt),
                             ms=ms, scanned=pq.last_stats["scanned"],
                             bytes=pq_only_bytes if refine is None else pq.memory_bytes() / n, build_s=build))
    log("ivf-pq done")

    h = HNSWIndex(dim, M=hnsw_M, ef_construction=hnsw_efc, seed=seed)
    t = time.perf_counter(); h.add(X); build = time.perf_counter() - t
    log(f"hnsw built in {build:.1f}s")
    for ef in (10, 20, 40, 80, 160):
        ms, ids = time_queries(h, Q, k, ef=ef)
        rows.append(dict(method=f"HNSW M={hnsw_M} efC={hnsw_efc}", setting=f"ef={ef}", recall=recall_at_k(ids, gt),
                         ms=ms, scanned=h.last_stats["scanned"], bytes=h.memory_bytes() / n, build_s=build))
    return rows


def to_markdown(rows) -> str:
    out = ["| method | setting | recall@k | ms/query | distance evals/query | bytes/vector | build (s) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['method']} | {r['setting']} | {r['recall']:.3f} | {r['ms']:.2f} | {r['scanned']:.0f} | "
                   f"{r['bytes']:.0f} | {r['build_s']:.1f} |")
    return "\n".join(out)
