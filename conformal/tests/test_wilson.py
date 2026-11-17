import numpy as np
import pytest
from conformal_pred import wilson_interval


def test_known_value():
    # 8/10 at z=1.96: textbook Wilson interval ~ (0.490, 0.943)
    lo, hi = wilson_interval([1] * 8 + [0] * 2)
    assert lo == pytest.approx(0.490, abs=2e-3) and hi == pytest.approx(0.943, abs=2e-3)


def test_extremes_stay_in_unit_interval():
    lo, hi = wilson_interval([1] * 20)
    assert hi == 1.0 and 0.8 < lo < 1.0
    lo, hi = wilson_interval([0] * 20)
    assert lo == 0.0 and 0 < hi < 0.2


def test_narrows_with_n_and_contains_mean():
    rng = np.random.default_rng(0)
    small = wilson_interval(rng.random(50) < 0.9)
    big = wilson_interval(rng.random(5000) < 0.9)
    assert (big[1] - big[0]) < (small[1] - small[0])
    assert big[0] < 0.9 < big[1]


def test_empty_raises():
    with pytest.raises(ValueError):
        wilson_interval([])
