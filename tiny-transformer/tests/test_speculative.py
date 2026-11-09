import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.speculative import speculative_generate  # noqa: E402


def _models(seed_t=1, seed_d=2):
    t = GPT(GPTConfig(vocab_size=5, block_size=16, n_layer=1, n_head=2, n_embd=16), seed=seed_t)
    d = GPT(GPTConfig(vocab_size=5, block_size=16, n_layer=1, n_head=2, n_embd=16), seed=seed_d)
    return t, d


def test_greedy_matches_target_greedy():
    t, d = _models()
    prompt = [1, 2]
    out, st = speculative_generate(t, d, prompt, 8, k=3, temperature=0)
    ref = t.generate(prompt, 8, temperature=0)
    assert out.tolist() == ref.tolist()
    assert st["target_calls"] <= 8


def test_identical_draft_accepts_everything():
    t, _ = _models()
    out, st = speculative_generate(t, t, [0], 9, k=4, temperature=1.0, seed=3)
    assert st["acceptance_rate"] == 1.0 and len(out) == 10
    assert st["tokens_per_target_call"] > 3


def test_preserves_target_distribution():
    t, d = _models()
    # bias the draft so it clearly differs from the target
    d.params["head_b"] = d.params["head_b"] + np.array([2, -2, 1, 0, -1], dtype=d.params["head_b"].dtype)
    prompt = [3]
    n = 3000
    spec = np.zeros(5)
    p0 = softmax_row(t, prompt)
    for s in range(n):
        out, _ = speculative_generate(t, d, prompt, 3, k=3, temperature=1.0, seed=s)
        spec[out[1]] += 1
    spec /= n
    assert np.abs(spec - p0).max() < 0.03  # ~4 sigma at n=3000


def softmax_row(m, ctx):
    lg = m.forward(np.array(ctx)[None])[0][0, -1].astype(np.float64)
    e = np.exp(lg - lg.max())
    return e / e.sum()


def test_validation():
    t, d = _models()
    with pytest.raises(ValueError):
        speculative_generate(t, d, [0], 3, k=0)
    other = GPT(GPTConfig(vocab_size=6, block_size=16, n_layer=1, n_head=2, n_embd=16))
    with pytest.raises(ValueError):
        speculative_generate(t, other, [0], 3)
