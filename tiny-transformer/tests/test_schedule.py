import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import Adam, warmup_cosine  # noqa: E402


def test_warmup_then_decay_endpoints():
    assert warmup_cosine(1, 100, 1.0, warmup=10) == 0.1
    assert warmup_cosine(10, 100, 1.0, warmup=10) == 1.0
    assert abs(warmup_cosine(100, 100, 1.0, warmup=10) - 0.1) < 1e-12


def test_monotone_decay_after_warmup():
    lrs = [warmup_cosine(s, 200, 3e-3, warmup=20) for s in range(20, 201)]
    assert all(a >= b - 1e-15 for a, b in zip(lrs, lrs[1:]))


def test_no_warmup_starts_at_peak():
    assert abs(warmup_cosine(1, 50, 2.0) - 2.0) < 0.01


def test_weight_decay_shrinks_matrices_not_biases():
    p = {"w": np.ones((2, 2)), "b": np.ones(2)}
    opt = Adam(p, lr=0.1, weight_decay=0.5)
    opt.step({"w": np.zeros((2, 2)), "b": np.zeros(2)})
    assert np.allclose(p["w"], 1 - 0.1 * 0.5)
    assert np.allclose(p["b"], 1.0)
