"""Compare two tokenizers on the same text: compression and boundary agreement."""
from __future__ import annotations


def _boundaries(tok, text: str) -> set[int]:
    """Byte offsets (in the UTF-8 encoding) where a token ends, excluding the end."""
    pos, out = 0, set()
    for i in tok.encode(text):
        pos += len(tok.vocab[i]) if i in tok.vocab else 0
        out.add(pos)
    out.discard(len(text.encode("utf-8")))
    return out


def compare(tok_a, tok_b, text: str) -> dict:
    """Return bytes, tokens and bytes/token for each tokenizer, plus the Jaccard
    overlap of their token-boundary sets (1.0 = identical segmentation) and
    `ratio` = tokens_a / tokens_b (<1 means A compresses better)."""
    nbytes = len(text.encode("utf-8"))
    ta, tb = tok_a.encode(text), tok_b.encode(text)
    ba, bb = _boundaries(tok_a, text), _boundaries(tok_b, text)
    union = ba | bb
    return {
        "bytes": nbytes,
        "tokens_a": len(ta), "tokens_b": len(tb),
        "bpt_a": nbytes / len(ta) if ta else 0.0,
        "bpt_b": nbytes / len(tb) if tb else 0.0,
        "ratio": len(ta) / len(tb) if tb else 0.0,
        "boundary_jaccard": len(ba & bb) / len(union) if union else 1.0,
    }
