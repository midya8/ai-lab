"""Token coverage curve: how much of a text the k most frequent tokens account for."""
from __future__ import annotations
from collections import Counter


def coverage(tok, text: str, ks=(10, 100, 1000)):
    """Encode `text`; for each k return the fraction of emitted tokens that are
    among the k most frequent distinct ids (ties broken by id), plus the number
    of distinct ids needed to cover 90% / 99% of emitted tokens.

    Returns {"tokens", "distinct", "at": {k: frac}, "k90", "k99"}."""
    ids = tok.encode(text)
    n = len(ids)
    ranked = sorted(Counter(ids).items(), key=lambda x: (-x[1], x[0]))
    cum, run = [], 0
    for _, v in ranked:
        run += v
        cum.append(run)

    def need(p):
        for i, c in enumerate(cum):
            if c >= p * n:
                return i + 1
        return 0

    at = {k: (cum[min(k, len(cum)) - 1] / n if n and cum else 0.0) for k in ks}
    return {"tokens": n, "distinct": len(ranked), "at": at, "k90": need(0.9), "k99": need(0.99)}
