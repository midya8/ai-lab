"""Active-learning review queue: pick the unlabeled tickets a human should label next.

Strategies: least-confidence, margin, entropy (uncertainty sampling) and random baseline.
`simulate` replays labeling with a budget and returns a learning curve per strategy.
"""
import random
import numpy as np
from .model import TicketClassifier

STRATEGIES = ("least_confidence", "margin", "entropy", "random")


def uncertainty(P, strategy):
    """Higher score = more uncertain = more worth labeling. P: (n, classes) probabilities."""
    P = np.asarray(P, dtype=float)
    if strategy == "least_confidence":
        return 1.0 - P.max(axis=1)
    if strategy == "margin":
        s = np.sort(P, axis=1)
        return 1.0 - (s[:, -1] - s[:, -2])
    if strategy == "entropy":
        return -(P * np.log(np.clip(P, 1e-12, 1))).sum(axis=1)
    raise ValueError(f"unknown strategy: {strategy}")


def select_batch(model, texts, k, strategy="margin", rng=None):
    """Indices (into texts) of the k tickets to review next, most valuable first."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    k = min(k, len(texts))
    if strategy == "random":
        return (rng or random.Random(0)).sample(range(len(texts)), k)
    scores = uncertainty(model.predict_proba(texts), strategy)
    return [int(i) for i in np.argsort(-scores, kind="stable")[:k]]


def review_queue(model, texts, k=10, strategy="margin"):
    """Human-readable queue: [{index, text, predicted, confidence, score}]."""
    P = model.predict_proba(texts)
    scores = uncertainty(P, strategy)
    out = []
    for i in select_batch(model, texts, k, strategy):
        j = int(P[i].argmax())
        out.append({"index": i, "text": texts[i], "predicted": model.classes[j],
                    "confidence": float(P[i, j]), "score": float(scores[i])})
    return out


def simulate(pool, test, strategy, seed_size=8, batch=4, rounds=6, seed=0):
    """Return [(n_labeled, accuracy)] as labels are revealed `batch` at a time.

    The seed set is stratified so every class appears (the model needs all classes)."""
    rng = random.Random(seed)
    by = {}
    for i, d in enumerate(pool):
        by.setdefault(d["category"], []).append(i)
    labeled = []
    for c in sorted(by):
        labeled.append(rng.choice(by[c]))
    while len(labeled) < seed_size:
        i = rng.randrange(len(pool))
        if i not in labeled:
            labeled.append(i)
    tt, tl = [d["text"] for d in test], [d["category"] for d in test]
    curve = []
    for r in range(rounds + 1):
        m = TicketClassifier().fit([pool[i]["text"] for i in labeled], [pool[i]["category"] for i in labeled])
        acc = sum(p == t for p, t in zip(m.predict(tt), tl)) / len(tl)
        curve.append((len(labeled), acc))
        if r == rounds:
            break
        rest = [i for i in range(len(pool)) if i not in set(labeled)]
        picks = select_batch(m, [pool[i]["text"] for i in rest], batch, strategy, rng)
        labeled += [rest[p] for p in picks]
    return curve


def compare(pool, test, strategies=STRATEGIES, seeds=(0, 1, 2), **kw):
    """Mean accuracy curve per strategy across seeds."""
    res = {}
    for s in strategies:
        curves = [simulate(pool, test, s, seed=sd, **kw) for sd in seeds]
        res[s] = [(curves[0][i][0], float(np.mean([c[i][1] for c in curves]))) for i in range(len(curves[0]))]
    return res
