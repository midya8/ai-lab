"""Unigram token entropy: how evenly a tokenizer spreads text over its vocab."""
from __future__ import annotations
import math
from collections import Counter


def token_entropy(tok, text: str):
    """Shannon entropy of the empirical token distribution of `text`.

    Returns tokens, distinct, entropy_bits (per token), bits_per_byte
    (unigram code length per UTF-8 byte; lower = better compressible under
    a unigram model) and efficiency (entropy / log2(distinct), 1.0 = uniform).
    This is an in-sample estimate, so it is optimistic for short texts."""
    ids = tok.encode(text)
    n = len(ids)
    if n == 0:
        return {"tokens": 0, "distinct": 0, "entropy_bits": 0.0,
                "bits_per_byte": 0.0, "efficiency": 0.0}
    c = Counter(ids)
    h = -sum(v / n * math.log2(v / n) for v in c.values())
    nbytes = len(text.encode())
    return {
        "tokens": n,
        "distinct": len(c),
        "entropy_bits": h,
        "bits_per_byte": h * n / nbytes if nbytes else 0.0,
        "efficiency": h / math.log2(len(c)) if len(c) > 1 else 1.0,
    }
