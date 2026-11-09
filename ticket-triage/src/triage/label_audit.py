"""Audit ticket category labels with datadoctor's confident-learning detector.

Support-ticket labels are typed by tired humans, so some are wrong. This module
turns tickets into TF-IDF features, gets out-of-fold probabilities, and ranks
tickets whose given category the model confidently disputes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "datadoctor", "src"))
from datadoctor.confident import find_label_issues, label_quality_scores  # noqa: E402
from datadoctor.cv import cross_val_probs  # noqa: E402
from datadoctor.models import SoftmaxRegression  # noqa: E402

from .model import TfidfVectorizer  # noqa: E402


def audit_labels(texts, labels, k=5, seed=0, top=None):
    """Return suspected label errors, most suspicious first.

    Each item: {index, text, given, suggested, quality}. Quality is the
    out-of-fold probability of the given label (low = suspicious).
    """
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    X = TfidfVectorizer().fit(texts).transform(texts)
    P = cross_val_probs(X, y, lambda: SoftmaxRegression(), k=k, seed=seed, n_classes=len(classes))
    mask = find_label_issues(P, y)
    q = label_quality_scores(P, y)
    idx = [i for i in np.argsort(q) if mask[i]]
    if top is not None:
        idx = idx[:top]
    return [{"index": int(i), "text": texts[i], "given": labels[i],
             "suggested": classes[int(P[i].argmax())], "quality": float(q[i])} for i in idx]


def flip_labels(labels, rate, seed=0):
    """Corrupt a fraction of labels uniformly to another class. Returns (noisy, flipped_indices)."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    n = int(round(rate * len(labels)))
    flipped = sorted(int(i) for i in rng.choice(len(labels), n, replace=False))
    noisy = list(labels)
    for i in flipped:
        noisy[i] = rng.choice([c for c in classes if c != labels[i]])
    return noisy, flipped
