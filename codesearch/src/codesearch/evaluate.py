"""Score retrieval against labelled queries: Recall@k, MRR, nDCG@10, with per-mode ablation."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .index import Index
from .search import MODES, Searcher


class LabelError(ValueError):
    """Labels refer to code that is not in the index (stale or misspelled)."""


@dataclass
class QueryCase:
    query: str
    relevant: list[str]
    note: str = ""


def load_queries(path: str | Path) -> tuple[dict, list[QueryCase]]:
    data = json.loads(Path(path).read_text())
    meta = {k: v for k, v in data.items() if k != "queries"}
    cases = []
    for i, q in enumerate(data["queries"]):
        if not q.get("query") or not q.get("relevant"):
            raise LabelError(f"query #{i} needs 'query' and a non-empty 'relevant' list")
        cases.append(QueryCase(q["query"], list(q["relevant"]), q.get("note", "")))
    return meta, cases


def check_labels(index: Index, cases: list[QueryCase]) -> None:
    missing = sorted({r for c in cases for r in c.relevant if r not in index.id_to_idx})
    if missing:
        raise LabelError("labels not found in the index: " + ", ".join(missing))


def dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


@dataclass
class ModeScore:
    mode: str
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    first_rank: list[int | None] = field(default_factory=list)  # per query, 1-based rank of first relevant

    def row(self) -> list[str]:
        return [self.mode, f"{self.recall_at_1:.3f}", f"{self.recall_at_5:.3f}", f"{self.recall_at_10:.3f}",
                f"{self.mrr:.3f}", f"{self.ndcg_at_10:.3f}"]


def score_mode(searcher: Searcher, cases: list[QueryCase], mode: str, depth: int = 10, **kw) -> ModeScore:
    r1 = r5 = r10 = mrr = ndcg = 0.0
    firsts: list[int | None] = []
    for c in cases:
        hits = searcher.search(c.query, k=depth, mode=mode, **kw)
        rel = set(c.relevant)
        gains = [1 if h.chunk.id in rel else 0 for h in hits]
        first = next((i + 1 for i, g in enumerate(gains) if g), None)
        firsts.append(first)
        if first:
            mrr += 1 / first
            r1 += first <= 1
            r5 += first <= 5
            r10 += first <= 10
        ideal = dcg([1] * min(len(rel), depth))
        ndcg += dcg(gains) / ideal if ideal else 0.0
    n = len(cases) or 1
    return ModeScore(mode, r1 / n, r5 / n, r10 / n, mrr / n, ndcg / n, firsts)


def run_eval(index: Index, cases: list[QueryCase], modes: tuple[str, ...] = MODES, **kw) -> list[ModeScore]:
    check_labels(index, cases)
    s = Searcher(index)
    return [score_mode(s, cases, m, **kw) for m in modes]


def format_table(scores: list[ModeScore]) -> str:
    head = ["mode", "R@1", "R@5", "R@10", "MRR", "nDCG@10"]
    rows = [head] + [s.row() for s in scores]
    w = [max(len(r[i]) for r in rows) for i in range(len(head))]
    fmt = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([fmt(rows[0]), "|" + "|".join("-" * (x + 2) for x in w) + "|"] + [fmt(r) for r in rows[1:]])


def misses(index: Index, cases: list[QueryCase], score: ModeScore, limit: int = 5) -> list[str]:
    """Human-readable list of queries whose first relevant hit was not in the top 3."""
    out = []
    for c, f in zip(cases, score.first_rank):
        if f is None or f > 3:
            out.append(f"[{'miss' if f is None else 'rank ' + str(f)}] {c.query}  ->  {c.relevant[0]}")
    return out[:limit] if limit else out
