"""TF-IDF + multinomial logistic regression (softmax) in NumPy, trained by gradient descent."""
import json
import re
import numpy as np

TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text):
    toks = TOKEN.findall(text.lower())
    return toks + [a + "_" + b for a, b in zip(toks, toks[1:])]


class TfidfVectorizer:
    def fit(self, texts):
        df = {}
        for t in texts:
            for w in set(tokenize(t)):
                df[w] = df.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if c >= 2))}
        n = len(texts)
        self.idf = np.zeros(len(self.vocab))
        for w, i in self.vocab.items():
            self.idf[i] = np.log((1 + n) / (1 + df[w])) + 1
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for r, t in enumerate(texts):
            for w in tokenize(t):
                i = self.vocab.get(w)
                if i is not None:
                    X[r, i] += 1
        X = np.log1p(X) * self.idf
        norm = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norm == 0, 1, norm)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TicketClassifier:
    def __init__(self, lr=1.0, epochs=300, l2=1e-4):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, texts, labels):
        self.vec = TfidfVectorizer().fit(texts)
        X = self.vec.transform(texts)
        self.classes = sorted(set(labels))
        y = np.array([self.classes.index(l) for l in labels])
        Y = np.eye(len(self.classes))[y]
        self.W = np.zeros((X.shape[1], len(self.classes)))
        self.b = np.zeros(len(self.classes))
        for _ in range(self.epochs):
            P = softmax(X @ self.W + self.b)
            G = (P - Y) / len(X)
            self.W -= self.lr * (X.T @ G + self.l2 * self.W)
            self.b -= self.lr * G.sum(axis=0)
        return self

    def predict_proba(self, texts):
        return softmax(self.vec.transform(texts) @ self.W + self.b)

    def predict(self, texts):
        return [self.classes[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def predict_with_confidence(self, text, threshold=0.5):
        p = self.predict_proba([text])[0]
        i = int(p.argmax())
        return {"label": self.classes[i], "confidence": float(p[i]), "needs_human": bool(p[i] < threshold)}


def confusion(y_true, y_pred, classes):
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[classes.index(t), classes.index(p)] += 1
    return m


def macro_f1(y_true, y_pred, classes):
    m = confusion(y_true, y_pred, classes)
    f = []
    for i in range(len(classes)):
        tp = m[i, i]
        p = tp / max(m[:, i].sum(), 1)
        r = tp / max(m[i].sum(), 1)
        f.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f))
