"""Multi-label tags: one-vs-rest logistic regression on shared TF-IDF features.

Tags (mobile, performance, security, enterprise) are orthogonal to the category,
so a ticket can carry zero or several. Each tag has its own decision threshold.
"""
import random
import numpy as np
from .data import generate
from .model import TfidfVectorizer
from .urgency import sigmoid

TAG_PHRASES = {
    "mobile": ["on my iPhone", "on Android", "on my tablet"],
    "performance": ["and it is extremely slow", "with huge latency", "it times out constantly"],
    "security": ["I suspect a data breach", "someone accessed my account without permission", "possible phishing attempt"],
    "enterprise": ["we are an enterprise customer", "our whole organization is affected", "per our SSO contract"],
}
TAGS = list(TAG_PHRASES)


def generate_tagged(n=400, seed=0, p=0.2):
    """Synthetic tickets with 0..4 independent tags appended as phrases."""
    rng = random.Random(seed + 7919)
    out = []
    for r in generate(n, seed):
        tags = [t for t in TAGS if rng.random() < p]
        text = r["text"] + "".join(" " + rng.choice(TAG_PHRASES[t]) for t in tags)
        out.append({**r, "text": text, "tags": tags})
    return out


def multilabel_metrics(y_true, y_pred):
    """y_*: (n, k) 0/1 arrays. Returns micro/macro F1, hamming loss, exact-match ratio."""
    yt, yp = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    def f1(t, p):
        tp = (t & p).sum()
        d = t.sum() + p.sum()
        return 0.0 if d == 0 else 2 * tp / d
    per = [f1(yt[:, j], yp[:, j]) for j in range(yt.shape[1])]
    return {"micro_f1": float(f1(yt.ravel(), yp.ravel())), "macro_f1": float(np.mean(per)),
            "per_tag_f1": [float(x) for x in per], "hamming_loss": float((yt != yp).mean()),
            "exact_match": float((yt == yp).all(axis=1).mean())}


class TagClassifier:
    def __init__(self, tags=TAGS, lr=2.0, epochs=400, l2=1e-4):
        self.tags, self.lr, self.epochs, self.l2 = list(tags), lr, epochs, l2

    def _Y(self, tag_lists):
        return np.array([[t in ts for t in self.tags] for ts in tag_lists], float)

    def fit(self, texts, tag_lists):
        self.vec = TfidfVectorizer().fit(texts)
        X, Y = self.vec.transform(texts), self._Y(tag_lists)
        k = len(self.tags)
        self.W, self.b = np.zeros((X.shape[1], k)), np.zeros(k)
        self.thresholds = np.full(k, 0.5)
        pos = np.maximum(Y.sum(0), 1)
        w = np.where(Y == 1, (len(Y) - Y.sum(0)) / pos, 1.0)  # balanced per tag
        w = w / w.mean(0)
        for _ in range(self.epochs):
            G = w * (sigmoid(X @ self.W + self.b) - Y) / len(Y)
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(0)
        return self

    def predict_proba(self, texts):
        return sigmoid(self.vec.transform(texts) @ self.W + self.b)

    def tune_thresholds(self, texts, tag_lists):
        """Per-tag threshold maximizing F1 on the given (validation) data."""
        P, Y = self.predict_proba(texts), self._Y(tag_lists).astype(bool)
        for j in range(len(self.tags)):
            best_f, best_t = -1.0, 0.5
            for t in np.unique(P[:, j]):
                p = P[:, j] >= t
                d = Y[:, j].sum() + p.sum()
                f = 0.0 if d == 0 else 2 * (Y[:, j] & p).sum() / d
                if f > best_f + 1e-12:
                    best_f, best_t = f, float(t)
            self.thresholds[j] = best_t
        return self.thresholds.copy()

    def predict(self, texts):
        return self.predict_proba(texts) >= self.thresholds

    def predict_tags(self, text):
        row = self.predict([text])[0]
        return [t for t, on in zip(self.tags, row) if on]
