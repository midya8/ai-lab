"""Adapter: use a trained microgrid MiniGPT as a specdec target or draft model.

Only the last `ctx` tokens are fed to the network (its positional table is that
long), so the model is a well-defined LM over any history length. Speculative
sampling stays exact for it, because target and draft are each a fixed function
of the history.
"""
import os
import sys
import numpy as np

_MG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "microgrid", "src")
if os.path.isdir(_MG) and _MG not in sys.path:
    sys.path.insert(0, _MG)


class MiniGPTLM:
    def __init__(self, model, ctx, temperature=1.0):
        assert temperature > 0
        self.model, self.ctx, self.T = model, ctx, temperature
        self.forward_calls = 0

    def probs(self, history):
        w = np.array([list(history)[-self.ctx:]])
        self.forward_calls += 1
        lg = self.model(w).data[0, -1].astype(float) / self.T
        p = np.exp(lg - lg.max())
        return p / p.sum()

    def probs_batch(self, histories):
        """Next-token probs for several histories using one forward pass per
        distinct window length (equal-length windows are stacked into a batch)."""
        wins = [list(h)[-self.ctx:] for h in histories]
        out = [None] * len(wins)
        by_len = {}
        for i, w in enumerate(wins):
            by_len.setdefault(len(w), []).append(i)
        for idxs in by_len.values():
            x = np.array([wins[i] for i in idxs])
            self.forward_calls += 1
            lg = self.model(x).data[:, -1].astype(float) / self.T
            e = np.exp(lg - lg.max(axis=1, keepdims=True))
            e /= e.sum(axis=1, keepdims=True)
            for j, i in enumerate(idxs):
                out[i] = e[j]
        return out


def n_params(model):
    return int(sum(p.data.size for p in model.parameters()))
