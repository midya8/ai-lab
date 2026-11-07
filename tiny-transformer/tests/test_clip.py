import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.clip import clip_grad_norm, global_norm  # noqa: E402


def test_global_norm_known_value():
    assert global_norm({"a": np.array([3.0]), "b": np.array([4.0])}) == pytest.approx(5.0)


def test_below_threshold_unchanged():
    g = {"a": np.array([0.3, 0.4])}
    out, n = clip_grad_norm(g, 1.0)
    assert n == pytest.approx(0.5)
    np.testing.assert_array_equal(out["a"], g["a"])


def test_clipped_norm_equals_max_and_direction_kept():
    g = {"a": np.array([3.0, 0.0]), "b": np.array([0.0, 4.0])}
    out, n = clip_grad_norm(g, 1.0)
    assert n == pytest.approx(5.0)
    assert global_norm(out) == pytest.approx(1.0, abs=1e-5)
    assert out["a"][0] / out["b"][1] == pytest.approx(0.75)


def test_dtype_preserved_and_invalid_max():
    g = {"a": np.full(4, 10.0, dtype=np.float32)}
    out, _ = clip_grad_norm(g, 1.0)
    assert out["a"].dtype == np.float32
    with pytest.raises(ValueError):
        clip_grad_norm(g, 0)


def test_real_model_grads_clip():
    m = GPT(GPTConfig(20, 8, 1, 2, 16), dtype=np.float64)
    rng = np.random.default_rng(1)
    x, y = rng.integers(0, 20, (4, 8)), rng.integers(0, 20, (4, 8))
    _, g = m.loss_and_grads(x, y)
    n = global_norm(g)
    out, pre = clip_grad_norm(g, n / 2)
    assert pre == pytest.approx(n)
    assert global_norm(out) == pytest.approx(n / 2, rel=1e-4)
