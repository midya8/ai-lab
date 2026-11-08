"""Learned complexity classifier.

A dependency-free logistic regression over hashed word/bigram features plus the
heuristic score as a prior feature. Train it on (prompt, difficulty) pairs where
difficulty is in [0, 1] (e.g. 1 = needed the top model, 0 = a cheap model sufficed)
and pass it to ``Router(scorer=clf.score)``. Reasons stay auditable: the top
contributing tokens are reported.
"""

from __future__ import annotations

import json
import math
import re
import zlib
from pathlib import Path

from .complexity import Complexity, score_prompt

_TOKEN = re.compile(r"[a-z0-9_']+")


def _features(prompt: str, dim: int) -> dict[int, float]:
    words = _TOKEN.findall(prompt.lower())
    grams = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
    feats: dict[int, float] = {}
    for g in grams:
        h = zlib.crc32(g.encode()) % dim
        feats[h] = feats.get(h, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in feats.values())) or 1.0
    return {k: v / norm for k, v in feats.items()}


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class LearnedComplexity:
    def __init__(self, dim: int = 4096, prior_weight: float = 0.5) -> None:
        self.dim = dim
        self.prior_weight = prior_weight  # blend toward the heuristic when untrained
        self.w = [0.0] * dim
        self.b = 0.0
        self.h_coef = 2.0                 # weight on the heuristic score feature
        self.trained = False

    def _z(self, prompt: str) -> tuple[float, dict[int, float], float]:
        f = _features(prompt, self.dim)
        h = score_prompt(prompt).score
        return self.b + self.h_coef * h + sum(self.w[i] * v for i, v in f.items()), f, h

    def predict(self, prompt: str) -> float:
        return _sigmoid(self._z(prompt)[0])

    def fit(self, data: list[tuple[str, float]], epochs: int = 60, lr: float = 0.5,
            l2: float = 1e-4) -> float:
        """SGD on log-loss; returns final mean loss. Deterministic (no shuffling)."""
        if not data:
            raise ValueError("no training data")
        for y in (d[1] for d in data):
            if not 0.0 <= y <= 1.0:
                raise ValueError("difficulty labels must be in [0, 1]")
        self.b = -self.h_coef * 0.5  # start with heuristic-centred bias
        for _ in range(epochs):
            for prompt, y in data:
                z, f, h = self._z(prompt)
                g = _sigmoid(z) - y
                for i, v in f.items():
                    self.w[i] -= lr * (g * v + l2 * self.w[i])
                self.b -= lr * g
                self.h_coef -= lr * g * h
        self.trained = True
        eps = 1e-9
        return -sum(y * math.log(self.predict(p) + eps) + (1 - y) * math.log(1 - self.predict(p) + eps)
                    for p, y in data) / len(data)

    def score(self, prompt: str) -> Complexity:
        base = score_prompt(prompt)
        if not self.trained:
            return base
        z, f, h = self._z(prompt)
        p = _sigmoid(z)
        top = sorted(f, key=lambda i: -abs(self.w[i] * f[i]))[:2]
        reasons = list(base.reasons) + [f"learned model p(hard)={p:.2f}"]
        if top and abs(self.w[top[0]] * f[top[0]]) > 0.05:
            reasons.append(f"{len(top)} hashed feature(s) drove the learned score")
        return Complexity(round(p, 4), tuple(reasons))

    def accuracy(self, data: list[tuple[str, float]], threshold: float = 0.5) -> float:
        ok = sum((self.predict(p) >= threshold) == (y >= threshold) for p, y in data)
        return ok / len(data)

    def save(self, path: str | Path) -> None:
        nz = {str(i): round(w, 6) for i, w in enumerate(self.w) if w}
        Path(path).write_text(json.dumps({"dim": self.dim, "b": self.b, "h_coef": self.h_coef, "w": nz}))

    @classmethod
    def load(cls, path: str | Path) -> "LearnedComplexity":
        d = json.loads(Path(path).read_text())
        m = cls(dim=d["dim"])
        m.b, m.h_coef, m.trained = d["b"], d["h_coef"], True
        for i, w in d["w"].items():
            m.w[int(i)] = w
        return m
