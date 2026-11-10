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


class TemperedLM:
    """Wrap a model and rescale its distribution: p**(1/T), renormalised.

    T>1 flattens (a vague draft), T<1 sharpens. Handy for building a draft that
    ranks tokens like the target but is under- or over-confident.
    """

    def __init__(self, base, temperature):
        assert temperature > 0
        self.base, self.T = base, temperature

    def probs(self, ctx):
        p = np.maximum(self.base.probs(ctx), 1e-12) ** (1.0 / self.T)
        return p / p.sum()


class MarkovLM:
    """First-order Markov chain with a given row-stochastic transition matrix."""

    def __init__(self, trans):
        self.trans = np.asarray(trans, dtype=float)

    def probs(self, ctx):
        return self.trans[ctx[-1]] if ctx else np.full(len(self.trans), 1.0 / len(self.trans))
