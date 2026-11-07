import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt.sampling import filter_logits, sample, softmax  # noqa: E402


def test_top_p_keeps_smallest_nucleus():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    lg = filter_logits(np.log(p), top_p=0.7)
    assert np.isfinite(lg).tolist() == [True, True, False, False]
    lg = filter_logits(np.log(p), top_p=0.79)
    assert np.isfinite(lg).tolist() == [True, True, False, False]
    lg = filter_logits(np.log(p), top_p=0.81)
    assert np.isfinite(lg).tolist() == [True, True, True, False]


def test_top_p_always_keeps_top_token():
    lg = filter_logits(np.log([0.9, 0.1]), top_p=0.01)
    assert np.isfinite(lg).tolist() == [True, False]


def test_top_p_none_or_one_is_noop():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(filter_logits(x, top_p=None), x)
    assert np.allclose(filter_logits(x, top_p=1.0), x)


def test_sampling_only_draws_from_nucleus():
    rng = np.random.default_rng(0)
    lg = np.log([0.5, 0.3, 0.15, 0.05])
    draws = {sample(lg, rng, top_p=0.7) for _ in range(300)}
    assert draws <= {0, 1} and draws == {0, 1}


def test_greedy_and_topk_combo():
    rng = np.random.default_rng(0)
    assert sample([0.1, 5, 0.2], rng, temperature=0) == 1
    assert all(sample([1, 2, 3, 4], rng, top_k=2, top_p=0.99) in (2, 3) for _ in range(50))
    assert abs(softmax(np.array([1.0, 1.0])).sum() - 1) < 1e-12
