"""Sensitivity of search quality to the knobs, on the DEV queries only.

    PYTHONPATH=src python experiments/sweeps.py

Prints RRF weights for the hybrid mode and per-field BM25F weights. The held-out file is
deliberately not touched here.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from codesearch.evaluate import load_queries, score_mode  # noqa: E402
from codesearch.index import DEFAULT_WEIGHTS, Index  # noqa: E402
from codesearch.search import Searcher  # noqa: E402
from codesearch.store import sync  # noqa: E402


def main() -> None:
    bench = HERE.parent / "bench"
    meta, cases = load_queries(bench / "queries_dev.json")
    chunks, _ = sync((bench / meta["root"]).resolve(), None, meta["include"])
    t = time.perf_counter()
    ix = Index.build(chunks)
    s = Searcher(ix)
    s.search("warm up the lsa", mode="lsa")
    print(f"{len(chunks)} chunks; index + LSA build {time.perf_counter() - t:.2f}s")

    t = time.perf_counter()
    for c in cases:
        s.search(c.query, mode="bm25f")
    print(f"bm25f query latency {1000 * (time.perf_counter() - t) / len(cases):.1f} ms/query\n")

    print("hybrid RRF weights (bm25f, lsa):   R@1   R@5   MRR")
    for w in [(1, 0), (1, 0.15), (1, 0.3), (1, 0.6), (1, 1.0)]:
        sc = score_mode(s, cases, "hybrid", weights=w)
        print(f"  {str(w):<28} {sc.recall_at_1:.3f} {sc.recall_at_5:.3f} {sc.mrr:.3f}")

    print("\nBM25F field weights (one change at a time from the defaults):")
    variants = {"defaults": {}, "name x0.5": {"name": 1.5}, "name x1.7": {"name": 5.0}, "doc x2": {"doc": 4.0},
                "body x0.5": {"body": 0.5}, "path off": {"path": 0.0}, "owner off": {"owner": 0.0}}
    for label, delta in variants.items():
        sc = score_mode(Searcher(Index.build(chunks, weights={**DEFAULT_WEIGHTS, **delta})), cases, "bm25f")
        print(f"  {label:<12} {sc.recall_at_1:.3f} {sc.recall_at_5:.3f} {sc.mrr:.3f}")


if __name__ == "__main__":
    main()
