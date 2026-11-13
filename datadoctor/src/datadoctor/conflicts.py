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
