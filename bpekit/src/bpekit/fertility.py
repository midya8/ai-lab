"""Fertility: tokens per word, a standard proxy for how well a tokenizer fits a text."""
from __future__ import annotations
from collections import Counter
import re

_WORD = re.compile(r"\w+", re.UNICODE)


def fertility(tok, text: str, worst: int = 5):
    """Encode each whitespace-delimited-ish word (regex \\w+) on its own.

    Returns dict: words, mean (tokens per word over all word occurrences),
    p95 (95th percentile of per-occurrence token count), single_frac (share of
    occurrences that are a single token) and worst (list of (word, tokens)
    for the distinct words needing the most tokens, ties broken by word)."""
    occ = Counter(_WORD.findall(text))
    n = sum(occ.values())
    if n == 0:
        return {"words": 0, "mean": 0.0, "p95": 0, "single_frac": 0.0, "worst": []}
    cost = {w: len(tok.encode(w)) for w in occ}
    counts = sorted(cost[w] for w, c in occ.items() for _ in range(c))
    p95 = counts[min(n - 1, int(0.95 * n))]
    single = sum(c for w, c in occ.items() if cost[w] == 1)
    top = sorted(cost.items(), key=lambda kv: (-kv[1], kv[0]))[:worst]
    return {"words": n, "mean": sum(counts) / n, "p95": p95,
            "single_frac": single / n, "worst": top}
