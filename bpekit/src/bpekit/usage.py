"""Vocabulary utilisation: which learned tokens actually get used on a text."""
from __future__ import annotations
from collections import Counter


def usage(tok, text: str, top: int = 10):
    """Encode `text` and summarise token usage.

    Returns a dict: tokens (count), unused (merged tokens never emitted),
    unused_frac, entropy_bits (unigram entropy of emitted ids) and
    top (list of (bytes, count)) for the most frequent merged tokens."""
    import math
    ids = tok.encode(text)
    c = Counter(ids)
    n = len(ids)
    merged = range(256, 256 + len(tok.merges))
    unused = [i for i in merged if c[i] == 0]
    ent = -sum(v / n * math.log2(v / n) for v in c.values()) if n else 0.0
    best = sorted(((i, v) for i, v in c.items() if i >= 256 and i in tok.vocab),
                  key=lambda x: (-x[1], x[0]))[:top]
    return {
        "tokens": n,
        "unused": len(unused),
        "unused_frac": len(unused) / max(1, len(tok.merges)),
        "entropy_bits": ent,
        "top": [(tok.vocab[i], v) for i, v in best],
    }
