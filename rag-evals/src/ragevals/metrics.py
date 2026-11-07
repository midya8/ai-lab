"""Retrieval metrics and an evaluation loop."""
from __future__ import annotations
import math
from typing import Callable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, r in enumerate(ranked, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, r in enumerate(ranked[:k], 1) if r in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate(search: Callable[[str, int], list[str]], dataset: list[dict], k: int = 5) -> dict:
    """dataset items: {"question": str, "relevant_docs": [doc_id,...]}. `search` returns ranked doc ids."""
    if not dataset:
        raise ValueError("empty dataset")
    rows = []
    for item in dataset:
        rel = set(item["relevant_docs"])
        ranked: list[str] = []
        for d in search(item["question"], k * 3):  # dedupe chunk hits into doc ranking
            if d not in ranked:
                ranked.append(d)
        rows.append((recall_at_k(ranked, rel, k), mrr(ranked, rel), ndcg_at_k(ranked, rel, k)))
    n = len(rows)
    return {"n": n, f"recall@{k}": sum(r[0] for r in rows) / n,
            "mrr": sum(r[1] for r in rows) / n, f"ndcg@{k}": sum(r[2] for r in rows) / n}
