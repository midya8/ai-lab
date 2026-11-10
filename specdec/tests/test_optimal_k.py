import numpy as np
import pytest
from specdec import optimal_k, expected_speedup, speculative_generate, NgramLM


def test_matches_bruteforce_and_monotone_in_alpha():
    ks = [optimal_k(a, 0.1)[0] for a in (0.3, 0.6, 0.9, 0.99)]
    assert ks == sorted(ks)
    k, s = optimal_k(0.8, 0.05)
    assert s == max(expected_speedup(0.8, kk, 0.05) for kk in range(1, 33))


def test_expensive_draft_prefers_short_k():
    assert optimal_k(0.7, 1.0)[0] == 1
    assert optimal_k(1.0, 0.0, k_max=10)[0] == 10


def test_invalid_inputs():
    with pytest.raises(ValueError):
        optimal_k(1.5, 0.1)
    with pytest.raises(ValueError):
        optimal_k(0.5, -1)


def test_expected_tokens_per_call_matches_simulation():
    r = np.random.default_rng(0)
    toks = list(r.integers(0, 4, 3000))
    t = NgramLM(4, 3).fit(toks)
    d = NgramLM(4, 1, alpha=1.0).fit(toks)
    st = {}
    n = 3000
    speculative_generate(t, d, [0], n, 4, np.random.default_rng(1), st)
    measured = n / st["target_calls"]
    alpha = st["accepted"] / max(st["proposed"], 1)
    predicted = (1 - alpha ** 5) / (1 - alpha)
    assert abs(measured - predicted) / predicted < 0.15
