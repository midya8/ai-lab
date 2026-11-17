import numpy as np
import pytest
from conformal_pred import exact_expected_coverage, calibration_size_for_slack, SplitConformal


def test_known_values():
    assert exact_expected_coverage(19, 0.1) == pytest.approx(0.9)       # ceil(20*.9)/20
    assert exact_expected_coverage(9, 0.1) == pytest.approx(0.9)        # ceil(10*.9)/10, exact
    assert exact_expected_coverage(5, 0.1) == 1.0                       # k>n: infinite interval
    assert exact_expected_coverage(10, 0.1) == pytest.approx(10 / 11)   # ceil(9.9)=10


def test_bounds_and_slack_search():
    for alpha in (0.05, 0.1, 0.2):
        for n in (20, 57, 200):
            c = exact_expected_coverage(n, alpha)
            assert 1 - alpha <= c < 1 - alpha + 1 / (n + 1) + 1e-12
    n = calibration_size_for_slack(0.1, 0.005)
    assert calibration_size_for_slack(0.1, 0.5) == 9
    assert exact_expected_coverage(n, 0.1) - 0.9 <= 0.005 + 1e-9
    assert n == 9  # 0.1 is a lucky alpha: 1/alpha integer -> exact at n=9
    n = calibration_size_for_slack(0.07, 0.002)
    assert exact_expected_coverage(n, 0.07) - 0.93 <= 0.002 + 1e-9
    assert all(exact_expected_coverage(m, 0.07) - 0.93 > 0.002 + 1e-9 for m in range(13, n))


def test_matches_simulation():
    rng = np.random.default_rng(1)
    n, alpha, hits, trials = 30, 0.1, 0, 4000
    for _ in range(trials):
        cal, t = rng.normal(size=n), rng.normal()
        sc = SplitConformal(lambda X: np.zeros(len(X)), alpha=alpha).calibrate(np.zeros(n), cal)
        lo, hi = sc.predict(np.zeros(1))
        hits += lo[0] <= t <= hi[0]
    assert hits / trials == pytest.approx(exact_expected_coverage(n, alpha), abs=0.02)


def test_errors():
    with pytest.raises(ValueError):
        exact_expected_coverage(0, 0.1)
    with pytest.raises(ValueError):
        calibration_size_for_slack(0.1, 0)
