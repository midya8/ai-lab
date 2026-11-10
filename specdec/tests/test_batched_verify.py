import numpy as np
from specdec.neural import MiniGPTLM
from specdec import speculative_generate
from microgrid import MiniGPT


def mk(seed, d, layers):
    return MiniGPTLM(MiniGPT(5, d=d, heads=2, layers=layers, ctx=6, seed=seed), 6)


class Unbatched:
    """Same model, probs_batch hidden, to compare against the batched path."""
    def __init__(self, m):
        self.m = m
    def probs(self, h):
        return self.m.probs(h)


def test_probs_batch_matches_single():
    m = mk(0, 8, 1)
    hs = [[1], [1, 2, 3], [0, 1, 2, 3, 4, 1, 2], [4, 4, 4, 4, 4, 4, 4, 3], [2, 2]]
    b = m.probs_batch(hs)
    for h, p in zip(hs, b):
        assert np.allclose(p, m.probs(h), atol=1e-5)


def test_same_output_and_fewer_forward_calls():
    d = mk(1, 4, 1)
    tb, tu = mk(0, 8, 1), Unbatched(mk(0, 8, 1))
    ob = speculative_generate(tb, d, [0, 1, 2, 3, 4, 0, 1], 30, 4, np.random.default_rng(3))
    ou = speculative_generate(tu, mk(1, 4, 1), [0, 1, 2, 3, 4, 0, 1], 30, 4, np.random.default_rng(3))
    assert ob == ou
    assert tb.forward_calls < tu.m.forward_calls
