"""Split-conformal prediction sets for the ticket classifier (LAC score).

Instead of one label plus a heuristic confidence threshold, return a *set* of
categories that contains the true one with probability >= 1 - alpha
(marginally, assuming calibration and test tickets are exchangeable).
Singleton sets can be auto-routed; larger sets go to a human with the short
list of candidates; empty sets mean "nothing plausible, escalate".
"""
import math
import numpy as np


def conformal_quantile(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest score; +inf if n is too small."""
    s = np.sort(np.asarray(scores, float))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


class ConformalTriage:
    def __init__(self, model, alpha=0.1):
        self.model, self.alpha, self.q = model, alpha, None

    def calibrate(self, texts, labels):
        """Calibrate on tickets NOT used to train `model`."""
        P = self.model.predict_proba(texts)
        idx = np.array([self.model.classes.index(l) for l in labels])
        self.q = conformal_quantile(1.0 - P[np.arange(len(idx)), idx], self.alpha)
        return self

    def predict_sets(self, texts):
        if self.q is None:
            raise RuntimeError("call calibrate() first")
        P = self.model.predict_proba(texts)
        return [[c for c, p in zip(self.model.classes, row) if 1.0 - p <= self.q] for row in P]

    def route(self, text):
        """auto: one category; review: several candidates; escalate: empty set."""
        s = self.predict_sets([text])[0]
        action = "auto" if len(s) == 1 else ("review" if s else "escalate")
        return {"action": action, "candidates": s}


def evaluate(sets, labels):
    n = len(labels)
    cov = sum(l in s for s, l in zip(sets, labels)) / n
    single = [s for s in sets if len(s) == 1]
    return {"coverage": cov, "mean_size": sum(map(len, sets)) / n,
            "auto_rate": len(single) / n, "empty_rate": sum(not s for s in sets) / n}
