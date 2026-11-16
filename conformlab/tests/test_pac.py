import numpy as np
import pytest
from conformlab import prob_coverage_below, pac_calibration_size, SplitConformal


def test_matches_simulation():
    n, alpha, target = 50, 0.1, 0.86
    rng = np.random.default_rng(0)
    # coverage of a split-conformal set = U_(k) for uniform scores
    import math
    k = math.ceil((n + 1) * (1 - alpha))
    cov = np.sort(rng.random((40000, n)), axis=1)[:, k - 1]
    assert prob_coverage_below(n, alpha, target) == pytest.approx((cov < target).mean(), abs=0.01)


def test_edges_and_monotone_target():
    assert prob_coverage_below(5, 0.1, 0.99) == 0.0          # k > n: infinite set
    assert prob_coverage_below(100, 0.1, 0.0) == 0.0
    a = prob_coverage_below(200, 0.1, 0.85)
    b = prob_coverage_below(200, 0.1, 0.89)
    assert a < b


def test_pac_size_is_minimal_and_valid():
    n = pac_calibration_size(0.1, 0.05, 0.05)
    assert prob_coverage_below(n, 0.1, 0.85) <= 0.05
    assert prob_coverage_below(n - 1, 0.1, 0.85) > 0.05
    assert n > 50  # far more than the marginal-guarantee size (9)
    assert pac_calibration_size(0.1, 0.02, 0.05) > n


def test_unreachable():
    with pytest.raises(ValueError):
        pac_calibration_size(0.1, 0.0001, 0.001, n_max=50)
