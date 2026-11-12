"""Vocabulary extension: keep an existing tokenizer's merges and learn more on new text.

Existing merge ids (256..256+n) are unchanged, so models trained on the old
vocabulary keep their meaning for every old token. Special tokens sit after
the merges, so they are renumbered (each shifts by the number of new merges).
"""
from __future__ import annotations
from collections import Counter
from .bpe import BPETokenizer, _SPLIT, _merge


def extend(tok: BPETokenizer, text: str, n_new: int, min_count: int = 2) -> BPETokenizer:
    """Return a tokenizer with up to `n_new` extra merges learned on `text`.

    Words are first segmented with the existing tokenizer, then the usual
    most-frequent-pair loop continues from there. Stops early when no pair
    occurs `min_count` times."""
    if n_new < 0:
        raise ValueError("n_new must be >= 0")
    words = Counter(_SPLIT.findall(text))
    seqs = {w: list(tok._encode_word(w)) for w in words}
    merges = list(tok.merges)
    for _ in range(n_new):
        stats = Counter()
        for w, s in seqs.items():
            for p in zip(s, s[1:]):
                stats[p] += words[w]
        if not stats:
            break
        best = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        if stats[best] < min_count:
            break
        new = 256 + len(merges)
        merges.append(best)
        for w in seqs:
            seqs[w] = _merge(seqs[w], best, new)
    specials = sorted(tok.specials, key=tok.specials.get)
    return BPETokenizer(merges, specials=specials)
