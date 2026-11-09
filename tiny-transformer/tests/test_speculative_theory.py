import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.speculative import expected_tokens_per_call, sweep_k  # noqa: E402


def test_closed_form_edges():
    assert expected_tokens_per_call(0.0, 5) == 1.0
    assert expected_tokens_per_call(1.0, 5) == 6.0
    assert expected_tokens_per_call(0.5, 1) == pytest.approx(1.5)
    assert expected_tokens_per_call(0.8, 4) == pytest.approx((1 - 0.8 ** 5) / 0.2)


def test_closed_form_validation():
    with pytest.raises(ValueError):
        expected_tokens_per_call(1.2, 3)
    with pytest.raises(ValueError):
        expected_tokens_per_call(0.5, 0)


def test_sweep_identical_draft_matches_prediction():
    t = GPT(GPTConfig(vocab_size=5, block_size=64, n_layer=1, n_head=2, n_embd=16), seed=1)
    rows = sweep_k(t, t, [1], ks=(2, 4), max_new_tokens=24)
    for r in rows:
        assert r["alpha"] == 1.0
        # last round is truncated to the token budget, so measured <= predicted
        assert r["measured"] <= r["predicted"] + 1e-9
        assert r["measured"] > 0.6 * r["predicted"]
