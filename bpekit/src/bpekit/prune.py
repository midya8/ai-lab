"""Vocabulary pruning: drop merges a reference corpus never emits.

A merge can be removed only if no kept merge depends on it (its output id is
not a component of a kept merge). Pruning is done to a fixed point, then ids
are renumbered. Encoding of the reference text is unchanged *as a byte
string round-trip*; token ids differ because of renumbering, and tokens
count may rise slightly when a removed merge was used only as a stepping
stone (those are kept, so on the reference text the token sequence length
is identical)."""
from __future__ import annotations
from collections import Counter
from .bpe import BPETokenizer


def prune(tok: BPETokenizer, text: str, min_count: int = 1) -> BPETokenizer:
    """Return a new tokenizer keeping merges emitted >= min_count times on
    `text`, plus any merges they transitively need as components."""
    c = Counter(tok.encode(text))
    keep = {i for i in range(256, 256 + len(tok.merges)) if c[i] >= min_count}
    stack = list(keep)
    while stack:
        i = stack.pop()
        for comp in tok.merges[i - 256]:
            if comp >= 256 and comp not in keep:
                keep.add(comp)
                stack.append(comp)
    order = sorted(keep)
    new_id = {i: i for i in range(256)}
    for n, i in enumerate(order):
        new_id[i] = 256 + n
    merges = [(new_id[a], new_id[b]) for a, b in (tok.merges[i - 256] for i in order)]
    specials = sorted(tok.specials, key=tok.specials.get)
    return BPETokenizer(merges, specials=specials)
