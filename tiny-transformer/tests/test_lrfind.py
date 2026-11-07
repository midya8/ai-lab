import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.lrfind import find_lr, lr_schedule, suggest_lr  # noqa: E402


def test_schedule_is_geometric_with_exact_endpoints():
    s = lr_schedule(1e-5, 1e-1, 5)
    assert s[0] == pytest.approx(1e-5) and s[-1] == pytest.approx(1e-1)
    ratios = [b / a for a, b in zip(s, s[1:])]
    assert ratios == pytest.approx([ratios[0]] * 4)


def test_schedule_validates():
    with pytest.raises(ValueError):
        lr_schedule(1.0, 0.1, 10)
    with pytest.raises(ValueError):
        lr_schedule(1e-3, 1.0, 1)


def test_suggest_picks_steepest_descent():
    lrs = [10 ** e for e in np.linspace(-6, 0, 13)]
    losses = [5, 5, 5, 5, 4.9, 3.0, 2.5, 2.4, 2.4, 3.0, 6.0, 12.0, 30.0]
    s = suggest_lr(lrs, losses)
    assert s == pytest.approx(lrs[5])


def test_find_lr_restores_weights_and_returns_curve():
    data = np.tile(np.arange(8), 40)
    model = GPT(GPTConfig(8, 8, 1, 2, 16))
    before = {k: v.copy() for k, v in model.params.items()}
    r = find_lr(model, data, batch=4, block=8, lr_min=1e-5, lr_max=1.0, steps=25)
    for k, v in before.items():
        np.testing.assert_array_equal(model.params[k], v)
    assert len(r["lrs"]) == len(r["losses"]) >= 3
    assert 1e-5 <= r["suggested"] <= 1.0
    assert r["losses"][min(len(r["losses"]) - 1, 15)] < r["losses"][0]
