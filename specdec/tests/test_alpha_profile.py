import numpy as np
import pytest
from specdec import NgramLM, alpha_profile, estimate_alpha, expected_speedup


def _pair():
    r = np.random.default_rng(1)
    toks = list(r.integers(0, 4, 2000))
    return NgramLM(4, 3).fit(toks), NgramLM(4, 1, alpha=1.0).fit(toks), toks


def test_mean_matches_estimate_alpha_and_bounds():
    t, d, toks = _pair()
    ctxs = [toks[i:i + 5] for i in range(0, 200, 5)]
    p = alpha_profile(t, d, ctxs, 0.1, 4)
    assert p["alpha_mean"] == pytest.approx(estimate_alpha(t, d, ctxs))
    assert 0 <= p["alpha_min"] <= p["alpha_mean"] <= p["alpha_max"] <= 1
    assert p["n"] == len(ctxs)


def test_identical_models_have_no_gap():
    t, _, toks = _pair()
    p = alpha_profile(t, t, [toks[:6], toks[10:16]], 0.1, 4)
    assert p["alpha_std"] == pytest.approx(0, abs=1e-9)
    assert p["speedup_from_mean_alpha"] == pytest.approx(expected_speedup(1.0, 4, 0.1))
    assert p["mean_of_context_speedups"] == pytest.approx(p["speedup_from_mean_alpha"])


def test_empty_raises():
    t, d, _ = _pair()
    with pytest.raises(ValueError):
        alpha_profile(t, d, [], 0.1, 4)
