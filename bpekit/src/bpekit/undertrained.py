"""Find merged tokens that a reference text emits rarely or never.

Such tokens are candidates for under-trained ("glitch") embeddings: a model
sees them too seldom to learn them. Counts come from encoding `text` only."""
from __future__ import annotations
from collections import Counter


def undertrained(tok, text: str, max_count: int = 0, limit: int | None = None):
    """Return [(id, token_bytes, count)] for merged ids emitted <= max_count
    times, rarest first (ties by id)."""
    c = Counter(tok.encode(text))
    out = [(i, tok.vocab[i], c[i]) for i in range(256, 256 + len(tok.merges))
           if i in tok.vocab and c[i] <= max_count]
    out.sort(key=lambda r: (r[2], r[0]))
    return out if limit is None else out[:limit]
