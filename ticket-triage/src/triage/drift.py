"""Drift monitor for a deployed ticket classifier.

Compares an incoming batch with a reference batch using signals that need no labels:
  * PSI (population stability index) of the predicted-category distribution
  * out-of-vocabulary (OOV) rate of the incoming tickets under the model's vocabulary
  * mean-confidence drop
Conventional PSI reading: <0.1 stable, 0.1-0.25 moderate, >0.25 major shift.
"""
import math
import numpy as np
from .model import tokenize


def psi(ref_counts, new_counts, eps=1e-4):
    """PSI between two count vectors over the same categories (eps-smoothed)."""
    r = np.asarray(ref_counts, dtype=float)
    n = np.asarray(new_counts, dtype=float)
    r = np.maximum(r / r.sum(), eps)
    n = np.maximum(n / n.sum(), eps)
    return float(np.sum((n - r) * np.log(n / r)))


def oov_rate(vec, texts):
    """Fraction of word tokens (unigrams and bigrams) not in the vectorizer vocabulary."""
    total = miss = 0
    for t in texts:
        for w in tokenize(t):
            total += 1
            miss += w not in vec.vocab
    return miss / total if total else 0.0


def severity(value):
    return "stable" if value < 0.1 else "moderate" if value < 0.25 else "major"


def drift_report(clf, reference_texts, new_texts):
    """clf: fitted TicketClassifier. Returns a dict of drift signals."""
    if not reference_texts or not new_texts:
        raise ValueError("need non-empty reference and new batches")
    pr, pn = clf.predict_proba(reference_texts), clf.predict_proba(new_texts)
    k = len(clf.classes)
    cr = np.bincount(pr.argmax(axis=1), minlength=k)
    cn = np.bincount(pn.argmax(axis=1), minlength=k)
    p = psi(cr, cn)
    o_ref, o_new = oov_rate(clf.vec, reference_texts), oov_rate(clf.vec, new_texts)
    c_ref, c_new = float(pr.max(axis=1).mean()), float(pn.max(axis=1).mean())
    return {
        "psi": p, "psi_severity": severity(p),
        "oov_ref": o_ref, "oov_new": o_new, "oov_delta": o_new - o_ref,
        "conf_ref": c_ref, "conf_new": c_new, "conf_drop": c_ref - c_new,
        "alert": p >= 0.25 or (o_new - o_ref) > 0.15 or (c_ref - c_new) > 0.15,
        "class_share_ref": dict(zip(clf.classes, (cr / cr.sum()).round(3).tolist())),
        "class_share_new": dict(zip(clf.classes, (cn / cn.sum()).round(3).tolist())),
    }
