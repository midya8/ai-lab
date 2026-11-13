"""Merge-tree inspection: how a token decomposes into the merges that built it."""
from __future__ import annotations


def merge_tree(tok, token_id: int):
    """Return the binary merge tree of `token_id` as nested tuples.

    A raw byte is a leaf `(id, None, None)`; a merged token is
    `(id, left_tree, right_tree)`. Raises ValueError for specials/unknown ids."""
    if token_id < 0 or token_id >= 256 + len(tok.merges):
        raise ValueError(f"not a byte or merged token id: {token_id}")
    if token_id < 256:
        return (token_id, None, None)
    a, b = tok.merges[token_id - 256]
    return (token_id, merge_tree(tok, a), merge_tree(tok, b))


def merge_depth(tok, token_id: int) -> int:
    """Number of merge levels beneath the token (0 for a raw byte)."""
    t = merge_tree(tok, token_id)

    def d(n):
        return 0 if n[1] is None else 1 + max(d(n[1]), d(n[2]))

    return d(t)


def format_tree(tok, token_id: int) -> str:
    """Indented text rendering, e.g. for debugging odd tokens."""
    lines = []

    def walk(n, indent):
        lines.append("  " * indent + repr(tok.vocab[n[0]].decode("utf-8", "replace")) + f" (id {n[0]})")
        if n[1] is not None:
            walk(n[1], indent + 1)
            walk(n[2], indent + 1)

    walk(merge_tree(tok, token_id), 0)
    return "\n".join(lines)
