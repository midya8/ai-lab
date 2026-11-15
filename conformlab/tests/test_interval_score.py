import numpy as np
from conformlab import interval_score


def test_hand_values():
    # covered: width 2; miss by 1 above: width 2 + (2/0.1)*1 = 22
    assert interval_score([0, 0], [2, 2], [1, 3], alpha=0.1) == (2 + 22) / 2


def test_proper_scoring_minimised_at_true_quantiles():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200_000)
    a = 0.1
    q = 1.6448536
    best = interval_score(np.full_like(y, -q), np.full_like(y, q), y, a)
    for s in (0.6, 0.8, 1.2, 1.5):
        assert interval_score(np.full_like(y, -q * s), np.full_like(y, q * s), y, a) > best


def test_width_alone_cannot_win():
    y = np.array([5.0, 5.0, 5.0])
    assert interval_score([0] * 3, [1] * 3, y) > interval_score([0] * 3, [10] * 3, y)
