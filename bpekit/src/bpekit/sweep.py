"""Vocab-size sweep: held-out bytes/token vs vocabulary size."""
from __future__ import annotations
from .bpe import train_fast


def sweep(text: str, vocab_sizes, holdout: float = 0.2):
    """Train on the head of `text` for each size; return rows of
    (vocab, merges_learned, train_bpt, heldout_bpt, heldout_gap)."""
    if not 0 < holdout < 1:
        raise ValueError("holdout must be in (0,1)")
    cut = int(len(text) * (1 - holdout))
    tr, ho = text[:cut], text[cut:]
    if not tr or not ho:
        raise ValueError("text too short")
    rows = []
    for v in sorted(set(vocab_sizes)):
        tok = train_fast(tr, v)
        a, b = tok.compression(tr), tok.compression(ho)
        rows.append((v, len(tok.merges) if hasattr(tok, "merges") else len(tok) - 256, a, b, a - b))
    return rows


def format_table(rows) -> str:
    out = ["vocab  merges  train_bpt  heldout_bpt  gap"]
    out += [f"{v:5d}  {m:6d}  {a:9.3f}  {b:11.3f}  {g:+.3f}" for v, m, a, b, g in rows]
    return "\n".join(out)
