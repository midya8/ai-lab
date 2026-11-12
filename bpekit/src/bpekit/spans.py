"""Token <-> text alignment and token-budget chunking."""
from __future__ import annotations
import re
from .bpe import _SPLIT


def encode_with_offsets(tok, text: str):
    """Return (ids, spans); spans[i] = (start, end) character offsets of token i in text.

    Special tokens map to their literal text. A token that covers only part of a
    multi-byte character reports the whole character, so spans of neighbours can
    overlap on such characters. text[s:e] joined over non-overlapping spans
    reconstructs the input for ordinary text.
    """
    ids, spans = [], []
    if tok.specials:
        pat = "(" + "|".join(re.escape(s) for s in tok.specials) + ")"
        parts = re.split(pat, text)
    else:
        parts = [text]
    pos = 0
    for part in parts:
        if not part:
            continue
        if part in tok.specials:
            ids.append(tok.specials[part]); spans.append((pos, pos + len(part)))
            pos += len(part)
            continue
        for w in _SPLIT.findall(part):
            # byte offset -> char offset table for this word
            b2c, bi = [], 0
            for ci, ch in enumerate(w):
                for _ in ch.encode("utf-8"):
                    b2c.append(ci)
            b = 0
            for t in tok._encode_word(w):
                n = len(tok.vocab[t])
                s, e = b2c[b], b2c[b + n - 1] + 1
                ids.append(t); spans.append((pos + s, pos + e))
                b += n
            pos += len(w)
    return ids, spans


def chunk_by_tokens(tok, text: str, max_tokens: int, overlap: int = 0):
    """Split text into chunks of at most max_tokens tokens, cutting only at
    pre-token (word) boundaries so no word is split. Consecutive chunks share
    about `overlap` tokens (rounded to whole words). A single word longer than
    max_tokens becomes its own oversized chunk. Returns a list of strings whose
    concatenation (overlap=0) equals the input."""
    if max_tokens < 1 or not 0 <= overlap < max_tokens:
        raise ValueError("need max_tokens >= 1 and 0 <= overlap < max_tokens")
    words = _SPLIT.findall(text)
    counts = [len(tok.encode(w)) for w in words]
    chunks, start = [], 0
    while start < len(words):
        end, total = start, 0
        while end < len(words) and (total + counts[end] <= max_tokens or end == start):
            total += counts[end]; end += 1
        chunks.append("".join(words[start:end]))
        if end >= len(words):
            break
        back, kept = end, 0
        while back > start + 1 and kept + counts[back - 1] <= overlap:
            back -= 1; kept += counts[back]
        start = back if back > start else end
    return chunks
