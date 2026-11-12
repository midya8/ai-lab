"""Surface-form consistency: does a word tokenize the same with and without a leading space?"""
from __future__ import annotations
import re


def boundary_variants(tok, text: str, min_count: int = 1, limit: int | None = None):
    """For each distinct alphabetic word in `text`, compare the token count of
    "word" (start of string) with " word" (mid-sentence), .

    Returns a list of dicts sorted by extra_tokens descending: word, count,
    bare (ids), spaced (ids), extra_tokens (len(spaced) - len(bare); 0 means
    the space is absorbed into a token, 1 means it cost a whole token). Only words
    seen at least `min_count` times are listed. Purely a tokenizer property;
    it measures no model."""
    counts: dict[str, int] = {}
    for w in re.findall(r"[A-Za-z]+", text):
        counts[w] = counts.get(w, 0) + 1
    rows = []
    for w, c in counts.items():
        if c < min_count:
            continue
        bare = tok.encode(w)
        spaced = tok.encode(" " + w)
        rows.append({"word": w, "count": c, "bare": bare, "spaced": spaced,
                     "extra_tokens": len(spaced) - len(bare)})
    rows.sort(key=lambda r: (-r["extra_tokens"], -r["count"], r["word"]))
    return rows[:limit] if limit is not None else rows
