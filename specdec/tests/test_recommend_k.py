import numpy as np
import pytest
from specdec import NgramLM, estimate_alpha, recommend_k, speculative_generate, optimal_k


def _models():
    r = np.random.default_rng(0)
    toks = [int(x) for x in r.integers(0, 4, 3000)]
    return toks, NgramLM(4, 3).fit(toks), NgramLM(4, 1, alpha=1.0).fit(toks)


def test_identical_models_alpha_one():
    toks, t, _ = _models()
    assert estimate_alpha(t, t, [toks[:5], toks[10:20]]) == pytest.approx(1.0)


def test_alpha_matches_measured_acceptance():
    toks, t, d = _models()
    ctxs = [toks[i:i + 6] for i in range(0, 600, 3)]
    a = estimate_alpha(t, d, ctxs)
    st = {}
    speculative_generate(t, d, [0], 4000, 4, np.random.default_rng(1), st)
    assert abs(a - st["accepted"] / st["proposed"]) < 0.1


def test_recommend_consistent_and_validates():
    toks, t, d = _models()
    ctxs = [toks[i:i + 6] for i in range(0, 300, 3)]
    k, a, s = recommend_k(t, d, ctxs, 0.1)
    assert (k, s) == optimal_k(a, 0.1)
    with pytest.raises(ValueError):
        estimate_alpha(t, d, [])
