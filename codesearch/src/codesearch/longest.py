"""Longest functions: line counts (docstring and blanks included) ranked, longest first."""
from __future__ import annotations


def longest(chunks, min_lines: int = 40, include_tests: bool = False):
    """Return [(chunk, n_lines)] for function/method chunks with >= min_lines, longest first."""
    out = []
    for c in chunks:
        if c.kind not in ("function", "method") or (c.is_test and not include_tests):
            continue
        n = c.end - c.start + 1
        if n >= min_lines:
            out.append((c, n))
    out.sort(key=lambda t: (-t[1], t[0].id))
    return out
