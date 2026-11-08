"""Chunk-size sweep: how retrieval quality changes with chunk size, per method."""
from __future__ import annotations
import json, pathlib
from typing import Sequence

from .cli import build_index, METHODS
from .metrics import evaluate


def sweep(docs_dir: str, dataset: list[dict], sizes: Sequence[int] = (15, 25, 40, 60, 100),
          methods: Sequence[str] = tuple(METHODS), k: int = 3) -> list[dict]:
    """Return one row per (chunk size, method) with metrics and the resulting chunk count."""
    rows = []
    for size in sizes:
        for m in methods:
            idx = build_index(docs_dir, max_words=size, method=m)
            res = evaluate(lambda q, kk, idx=idx: [c.doc_id for c, _ in idx.search(q, kk)], dataset, k)
            rows.append({"max_words": size, "method": m, "chunks": len(idx.chunks), **res})
    return rows


def best(rows: list[dict], key: str = "mrr") -> dict:
    """Best row by `key`; ties go to the larger chunk size (fewer chunks, cheaper index)."""
    return max(rows, key=lambda r: (r[key], r["max_words"]))


def to_markdown(rows: list[dict], k: int = 3) -> str:
    rk = f"recall@{k}"
    out = [f"| max_words | method | chunks | {rk} | MRR |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['max_words']} | {r['method']} | {r['chunks']} | {r[rk]:.3f} | {r['mrr']:.3f} |")
    return "\n".join(out)
