import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402


def make():
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=8)
    m = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    return m, rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))


def test_zero_smoothing_unchanged():
    m, x, y = make()
    l0, _ = m.loss_and_grads(x, y)
    assert abs(l0 - m.forward(x, y)[1]) < 1e-12


def test_smoothed_loss_formula():
    m, x, y = make()
    nll = m.forward(x, y)[1]
    logits = m.forward(x)[0]
    z = logits - logits.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    m.label_smoothing = 0.1
    l, _ = m.loss_and_grads(x, y)
    assert abs(l - (0.9 * nll - 0.1 * lp.mean())) < 1e-9
    assert m.forward(x, y)[1] == nll  # eval path unaffected


def test_grads_match_finite_differences():
    m, x, y = make()
    m.label_smoothing = 0.2
    _, G = m.loss_and_grads(x, y)
    rng = np.random.default_rng(0)
    for k in ["wte", "head_W", "head_b"]:
        for _ in range(4):
            i = tuple(rng.integers(0, s) for s in m.params[k].shape)
            old = m.params[k][i]; h = 1e-5
            m.params[k][i] = old + h; lp = m.loss_and_grads(x, y)[0]
            m.params[k][i] = old - h; lm = m.loss_and_grads(x, y)[0]
            m.params[k][i] = old
            assert abs((lp - lm) / (2 * h) - G[k][i]) < 1e-6
