import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt.speculative import best_k, expected_speedup, expected_tokens_per_call  # noqa: E402


def test_free_draft_equals_tokens_per_call():
    assert expected_speedup(0.7, 4, 0.0) == pytest.approx(expected_tokens_per_call(0.7, 4))


def test_expensive_or_bad_draft_is_a_slowdown():
    assert expected_speedup(0.1, 4, 0.5) < 1.0
    assert expected_speedup(0.9, 4, 1.0) < 1.0


def test_best_k_grows_with_alpha_and_shrinks_with_cost():
    assert best_k(0.95, 0.05)[0] > best_k(0.6, 0.05)[0]
    assert best_k(0.8, 0.02)[0] >= best_k(0.8, 0.3)[0]


def test_best_k_is_argmax_and_validates():
    k, s = best_k(0.8, 0.1)
    assert all(expected_speedup(0.8, j, 0.1) <= s + 1e-12 for j in range(1, 17))
    assert k >= 1
    with pytest.raises(ValueError):
        expected_speedup(0.5, 2, -1)
