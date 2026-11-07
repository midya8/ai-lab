"""Urgency head: binary logistic regression on the shared TF-IDF features.

Urgent tickets are the minority class, so the loss can be class-weighted and the
decision threshold tuned for recall (missing an urgent ticket costs more than a
false alarm).
"""
import numpy as np
from .model import TicketClassifier, TfidfVectorizer


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def precision_recall_f1(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum())
    p = tp / max(int(y_pred.sum()), 1)
    r = tp / max(int(y_true.sum()), 1)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"precision": p, "recall": r, "f1": f}


class UrgencyClassifier:
    def __init__(self, lr=2.0, epochs=400, l2=1e-4, pos_weight=None, threshold=0.5):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.pos_weight, self.threshold = pos_weight, threshold  # pos_weight None -> balanced

    def fit(self, texts, urgent):
        self.vec = TfidfVectorizer().fit(texts)
        X = self.vec.transform(texts)
        y = np.asarray(urgent, float)
        pos = max(y.sum(), 1)
        w_pos = self.pos_weight if self.pos_weight is not None else (len(y) - y.sum()) / pos
        w = np.where(y == 1, w_pos, 1.0)
        w = w / w.mean()
        self.W = np.zeros(X.shape[1])
        self.b = 0.0
        for _ in range(self.epochs):
            g = w * (sigmoid(X @ self.W + self.b) - y) / len(y)
            self.W -= self.lr * (X.T @ g + self.l2 * self.W)
            self.b -= self.lr * g.sum()
        return self

    def predict_proba(self, texts):
        return sigmoid(self.vec.transform(texts) @ self.W + self.b)

    def predict(self, texts):
        return self.predict_proba(texts) >= self.threshold

    def tune_threshold(self, texts, urgent, min_recall=0.95):
        """Pick the highest threshold whose recall on this data is >= min_recall."""
        p = self.predict_proba(texts)
        y = np.asarray(urgent, bool)
        best = float(p.min())
        for t in np.unique(p)[::-1]:
            if precision_recall_f1(y, p >= t)["recall"] >= min_recall:
                best = float(t)
                break
        self.threshold = best
        return best


class Triage:
    """Category classifier + urgency head, with a combined routing decision."""

    def __init__(self, min_confidence=0.5):
        self.min_confidence = min_confidence

    def fit(self, records):
        texts = [r["text"] for r in records]
        self.category = TicketClassifier().fit(texts, [r["category"] for r in records])
        self.urgency = UrgencyClassifier().fit(texts, [r["urgent"] for r in records])
        return self

    def route(self, text):
        c = self.category.predict_with_confidence(text, self.min_confidence)
        u = float(self.urgency.predict_proba([text])[0])
        urgent = u >= self.urgency.threshold
        queue = f"{c['label']}:urgent" if urgent else c["label"]
        return {**c, "urgency_prob": u, "urgent": urgent, "queue": queue,
                "needs_human": c["needs_human"] or urgent}
