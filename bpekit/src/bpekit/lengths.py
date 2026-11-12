"""Token-length distribution: how many bytes each emitted token covers."""
from __future__ import annotations
from collections import Counter


def length_hist(tok, text: str):
    """Encode `text` and histogram the byte length of emitted tokens.

    Returns a dict: tokens, hist ({byte_len: count}), byte_fallback_frac
    (share of tokens that are single raw bytes), mean and max length.
    Specials count by their UTF-8 length."""
    ids = tok.encode(text)
    lens = Counter(len(tok.vocab[i]) if i in tok.vocab else 0 for i in ids)
    n = len(ids)
    total = sum(k * v for k, v in lens.items())
    return {
        "tokens": n,
        "hist": dict(sorted(lens.items())),
        "byte_fallback_frac": lens.get(1, 0) / n if n else 0.0,
        "mean": total / n if n else 0.0,
        "max": max(lens) if lens else 0,
    }
