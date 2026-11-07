import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.accum import accumulate_grads  # noqa: E402


def _setup():
    m = GPT(GPTConfig(20, 8, 1, 2, 16), dtype=np.float64)
    rng = np.random.default_rng(1)
    x = rng.integers(0, 20, (8, 8))
    y = rng.integers(0, 20, (8, 8))
    return m, x, y


def test_matches_full_batch():
    m, x, y = _setup()
    l0, g0 = m.loss_and_grads(x, y)
    l1, g1 = accumulate_grads(m, x, y, 2)
    assert abs(l0 - l1) < 1e-9
    for k in g0:
        assert np.allclose(g0[k], g1[k], atol=1e-9), k


def test_single_micro_batch_is_identity():
    m, x, y = _setup()
    l0, g0 = m.loss_and_grads(x, y)
    l1, g1 = accumulate_grads(m, x, y, 8)
    assert l0 == pytest.approx(l1)
    assert all(np.allclose(g0[k], g1[k]) for k in g0)


def test_indivisible_raises():
    m, x, y = _setup()
    with pytest.raises(ValueError):
        accumulate_grads(m, x, y, 3)
