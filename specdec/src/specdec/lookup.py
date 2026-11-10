"""Prompt-lookup draft: a model-free draft distribution built from the context.

If the last `n` tokens occurred earlier in the context, the token that followed
the most recent earlier occurrence gets mass `conf`; the rest is uniform.
With no match the draft is uniform. It is a valid q(.|ctx) that depends only on
ctx, so speculative_generate stays exactly target-distributed; it only affects
the acceptance rate. Costs ~nothing, so it suits copy-heavy text (code, quotes).
"""
import numpy as np


class PromptLookupLM:
    def __init__(self, vocab, n=2, conf=0.9):
        assert vocab > 1 and n >= 1 and 0 < conf < 1
        self.V, self.n, self.conf = vocab, n, conf

    def probs(self, ctx):
        q = np.full(self.V, 1.0 / self.V)
        n = self.n
        if len(ctx) > n:
            key = list(ctx[-n:])
            for i in range(len(ctx) - n - 1, -1, -1):
                if list(ctx[i:i + n]) == key:
                    q = np.full(self.V, (1 - self.conf) / self.V)
                    q[ctx[i + n]] += self.conf
                    break
        return q
