"""A small multinomial logistic regression (NumPy, full-batch Adam, L2)."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxRegression:
    """Standardises inputs internally; ``n_classes`` may exceed the classes seen in ``fit``.

    Cross-validation folds can miss a rare class entirely, so the output always has
    ``n_classes`` columns and an unseen class simply gets a very small probability.
    """

    def __init__(self, l2: float = 1e-2, lr: float = 0.1, epochs: int = 300, seed: int = 0):
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "SoftmaxRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        k = int(n_classes if n_classes is not None else y.max() + 1)
        n, d = X.shape
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        Z = (X - self.mu) / self.sd
        Y = np.zeros((n, k))
        Y[np.arange(n), y] = 1.0
        W = np.zeros((d, k))
        b = np.zeros(k)
        # classes absent from this training set start (and stay) strongly negative
        present = np.bincount(y, minlength=k) > 0
        b[~present] = -10.0
        m = [np.zeros_like(W), np.zeros_like(b)]
        v = [np.zeros_like(W), np.zeros_like(b)]
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            P = softmax(Z @ W + b)
            G = (P - Y) / n
            gW = Z.T @ G + self.l2 * W
            gb = G.sum(axis=0)
            gb[~present] = 0.0
            for i, (p, g) in enumerate(((W, gW), (b, gb))):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                p -= self.lr * (m[i] / (1 - b1**t)) / (np.sqrt(v[i] / (1 - b2**t)) + eps)
        self.W, self.b = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None, "fit first"
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return softmax(Z @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


class KNNClassifier:
    """Distance-weighted k-NN probabilities (Laplace-smoothed). A non-linear, model-agnostic alternative
    to :class:`SoftmaxRegression` for out-of-sample probabilities."""

    def __init__(self, k: int = 15, seed: int = 0):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "KNNClassifier":
        X = np.asarray(X, dtype=float)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        self.Z = (X - self.mu) / self.sd
        self.y = np.asarray(y, dtype=int)
        self.K = int(n_classes if n_classes is not None else self.y.max() + 1)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Q = (np.asarray(X, dtype=float) - self.mu) / self.sd
        k = min(self.k, len(self.Z))
        sq = (self.Z * self.Z).sum(1)
        out = np.zeros((len(Q), self.K))
        for s in range(0, len(Q), 512):
            blk = Q[s : s + 512]
            D = np.maximum((blk * blk).sum(1)[:, None] + sq[None, :] - 2 * blk @ self.Z.T, 0)
            nn = np.argpartition(D, k - 1, axis=1)[:, :k]
            for r in range(len(blk)):
                np.add.at(out[s + r], self.y[nn[r]], 1.0)
        out += 0.5  # Laplace smoothing keeps probabilities off exactly 0
        return out / out.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
