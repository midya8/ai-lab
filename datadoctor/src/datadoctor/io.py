"""CSV loading: numeric feature columns, a label column, optional text column."""

from __future__ import annotations

import csv

import numpy as np


def load_csv(path: str, label: str, text: str | None = None):
    """Return (X, y, class_names, texts, feature_names).

    Features are every column that parses as a number in all rows (the label and text
    columns are excluded). Labels may be strings; they are mapped to 0..K-1 sorted.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty CSV")
    if label not in rows[0]:
        raise ValueError(f"label column {label!r} not in {list(rows[0])}")
    feats = []
    for col in rows[0]:
        if col in (label, text):
            continue
        try:
            [float(r[col]) for r in rows]
        except (TypeError, ValueError):
            continue
        feats.append(col)
    if not feats:
        raise ValueError("no numeric feature columns found")
    X = np.array([[float(r[c]) for c in feats] for r in rows])
    raw = [r[label] for r in rows]
    names = sorted(set(raw))
    y = np.array([names.index(v) for v in raw])
    texts = [r[text] for r in rows] if text else None
    return X, y, names, texts, feats


def write_csv_without(path_in: str, path_out: str, drop_rows: set[int]) -> int:
    """Copy a CSV minus the given 0-based data rows. Returns rows written."""
    with open(path_in, newline="") as f, open(path_out, "w", newline="") as g:
        r = csv.reader(f)
        w = csv.writer(g)
        w.writerow(next(r))
        n = 0
        for i, row in enumerate(r):
            if i not in drop_rows:
                w.writerow(row)
                n += 1
    return n
