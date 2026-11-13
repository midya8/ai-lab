"""Conflicting labels: identical feature rows that carry different labels."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def label_conflicts(X, y, decimals: int = 6) -> dict:
    """Group rows with identical features (rounded to `decimals`) and report groups with >1 label.

    Returns {"n_groups", "n_rows", "rate", "groups": [{"rows", "labels", "majority", "minority_rows"}]}.
    Minority rows are the likeliest label errors (ties: no minority is named, all rows listed).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and match len(y)")
    buckets = defaultdict(list)
    for i, row in enumerate(np.round(X, decimals)):
        buckets[row.tobytes()].append(i)
    groups = []
    for rows in buckets.values():
        labs = y[rows]
        uniq, cnt = np.unique(labs, return_counts=True)
        if len(uniq) < 2:
            continue
        top = cnt.max()
        tie = (cnt == top).sum() > 1
        majority = None if tie else uniq[cnt.argmax()].item()
        minority = [] if tie else [r for r in rows if y[r] != uniq[cnt.argmax()]]
        groups.append({"rows": rows, "labels": [u.item() for u in uniq],
                       "majority": majority, "minority_rows": minority})
    groups.sort(key=lambda g: -len(g["rows"]))
    n_rows = sum(len(g["rows"]) for g in groups)
    return {"n_groups": len(groups), "n_rows": n_rows,
            "rate": n_rows / len(y) if len(y) else 0.0, "groups": groups}


def near_conflicts(X, y, radius: float, max_rows: int = 5000) -> dict:
    """Pairs of rows closer than `radius` (Euclidean) that carry different labels.

    Extends `label_conflicts` from exact duplicates to near-duplicates. Exact O(n^2) scan in
    blocks, so `max_rows` guards against accidental huge inputs.
    Returns {"n_pairs", "pairs": [(i, j, distance)] sorted by distance, "rows": sorted row ids involved}.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2-D and match len(y)")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    n = len(X)
    if n > max_rows:
        raise ValueError(f"{n} rows exceeds max_rows={max_rows}")
    sq = (X ** 2).sum(1)
    pairs = []
    block = 512
    for s in range(0, n, block):
        d2 = sq[s:s + block, None] + sq[None, :] - 2 * X[s:s + block] @ X.T
        np.maximum(d2, 0, out=d2)
        ii, jj = np.nonzero((d2 <= radius ** 2) & (y[s:s + block, None] != y[None, :]))
        for a, b in zip(ii, jj):
            i = int(a) + s
            if i < b:
                pairs.append((i, int(b), float(np.sqrt(d2[a, b]))))
    pairs.sort(key=lambda p: p[2])
    rows = sorted({r for p in pairs for r in p[:2]})
    return {"n_pairs": len(pairs), "pairs": pairs, "rows": rows}
