from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class(cm):
    """Precision / recall / F1 per class (0 when undefined)."""
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1


def accuracy(y_true, y_pred):
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def format_report(cm, names):
    prec, rec, f1 = per_class(cm)
    w = max(len(n) for n in names)
    lines = ["confusion (rows=true, cols=pred)", " " * (w + 1) + " ".join(f"{n[:5]:>5}" for n in names)]
    for n, row in zip(names, cm):
        lines.append(f"{n:<{w}} " + " ".join(f"{v:>5d}" for v in row))
    lines.append("")
    lines.append(f"{'class':<{w}}  prec   rec    f1")
    for n, p, r, f in zip(names, prec, rec, f1):
        lines.append(f"{n:<{w}}  {p:.3f}  {r:.3f}  {f:.3f}")
    return "\n".join(lines)
