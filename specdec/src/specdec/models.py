"""Tiny count-based language models used as target/draft pairs."""
import numpy as np


class NgramLM:
    """Add-alpha smoothed n-gram LM over integer tokens. probs(ctx) -> vector."""

    def __init__(self, vocab_size, order=3, alpha=0.1):
        self.V, self.order, self.alpha = vocab_size, order, alpha
        self.counts = {}

    def fit(self, tokens):
        n = self.order - 1
        for i in range(len(tokens)):
            ctx = tuple(tokens[max(0, i - n):i]) if n else ()
            row = self.counts.setdefault(ctx, np.zeros(self.V))
            row[tokens[i]] += 1
        return self

    def probs(self, ctx):
        n = self.order - 1
        key = tuple(ctx[-n:]) if n else ()
        row = self.counts.get(key)
        row = np.zeros(self.V) if row is None else row
        p = row + self.alpha
        return p / p.sum()


class MixtureLM:
    """Convex mix of two models: handy for a draft with controllable quality."""

    def __init__(self, a, b, w):
        self.a, self.b, self.w = a, b, w

    def probs(self, ctx):
        return self.w * self.a.probs(ctx) + (1 - self.w) * self.b.probs(ctx)
