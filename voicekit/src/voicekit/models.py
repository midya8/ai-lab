"""Two small classifiers, both from scratch: an MLP (manual backprop, Adam) and DTW 1-NN."""
from __future__ import annotations

import numpy as np


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class MLP:
    def __init__(self, hidden: int = 96, epochs: int = 80, lr: float = 3e-3, weight_decay: float = 1e-4,
                 batch: int = 64, dropout: float = 0.2, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.wd = hidden, epochs, lr, weight_decay
        self.batch, self.dropout, self.seed = batch, dropout, seed
        self.params: dict[str, np.ndarray] | None = None
        self.mu = self.sd = None
        self.loss_history: list[float] = []

    def _forward(self, X, train=False, rng=None):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        h = np.maximum(z1, 0)
        mask = None
        if train and self.dropout > 0:
            mask = (rng.random(h.shape) >= self.dropout) / (1 - self.dropout)
            h = h * mask
        return z1, h, mask, h @ p["W2"] + p["b2"]

    def fit(self, X: np.ndarray, y: np.ndarray, n_classes: int | None = None) -> "MLP":
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        k = n_classes or int(y.max()) + 1
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        rng = np.random.default_rng(self.seed)
        d = Xn.shape[1]
        self.params = {
            "W1": rng.standard_normal((d, self.hidden)) * np.sqrt(2.0 / d), "b1": np.zeros(self.hidden),
            "W2": rng.standard_normal((self.hidden, k)) * np.sqrt(2.0 / self.hidden), "b2": np.zeros(k),
        }
        m = {n: np.zeros_like(v) for n, v in self.params.items()}
        v2 = {n: np.zeros_like(v) for n, v in self.params.items()}
        step = 0
        n = len(Xn)
        Y = np.eye(k)[y]
        self.loss_history = []
        for _ in range(self.epochs):
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, self.batch):
                idx = order[s:s + self.batch]
                xb, yb = Xn[idx], Y[idx]
                z1, h, mask, logits = self._forward(xb, True, rng)
                pr = _softmax(logits)
                total += float(-np.log(pr[np.arange(len(idx)), y[idx]] + 1e-12).sum())
                g = (pr - yb) / len(idx)
                grads = {"W2": h.T @ g, "b2": g.sum(0)}
                gh = g @ self.params["W2"].T
                if mask is not None:
                    gh = gh * mask
                gz = gh * (z1 > 0)
                grads["W1"], grads["b1"] = xb.T @ gz, gz.sum(0)
                step += 1
                for name, grad in grads.items():
                    if name.startswith("W"):
                        grad = grad + self.wd * self.params[name]
                    m[name] = 0.9 * m[name] + 0.1 * grad
                    v2[name] = 0.999 * v2[name] + 0.001 * grad**2
                    mh, vh = m[name] / (1 - 0.9**step), v2[name] / (1 - 0.999**step)
                    self.params[name] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
            self.loss_history.append(total / n)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("model is not fitted")
        Xn = (np.atleast_2d(np.asarray(X, float)) - self.mu) / self.sd
        return _softmax(self._forward(Xn)[3])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(1)

    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu, sd=self.sd, **self.params)

    @classmethod
    def load(cls, path: str) -> "MLP":
        z = np.load(path)
        obj = cls(hidden=z["W1"].shape[1])
        obj.mu, obj.sd = z["mu"], z["sd"]
        obj.params = {k: z[k] for k in ("W1", "b1", "W2", "b2")}
        return obj


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = 0.35) -> float:
    """Length-normalised DTW distance with a Sakoe-Chiba band (rows are frames)."""
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    w = max(int(band * max(n, m)), abs(n - m) + 1)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


class DTWClassifier:
    """1-NN over a few stored templates per class, matched by DTW."""

    def __init__(self, per_class: int = 5, seed: int = 0):
        self.per_class, self.seed = per_class, seed
        self.templates: list[tuple[int, np.ndarray]] = []

    def fit(self, seqs: list[np.ndarray], y) -> "DTWClassifier":
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        self.templates = []
        for c in sorted(set(y.tolist())):
            idx = np.flatnonzero(y == c)
            for i in rng.choice(idx, size=min(self.per_class, len(idx)), replace=False):
                self.templates.append((int(c), seqs[i]))
        return self

    def class_distances(self, seq: np.ndarray, n_classes: int) -> np.ndarray:
        best = np.full(n_classes, np.inf)
        for c, t in self.templates:
            best[c] = min(best[c], dtw_distance(seq, t))
        return best

    def predict(self, seq: np.ndarray, n_classes: int) -> int:
        return int(self.class_distances(seq, n_classes).argmin())
