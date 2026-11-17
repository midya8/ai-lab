import numpy as np
from conformal_pred import coverage_calibration_curve, hetero_regression, mean_true


def test_valid_across_alphas():
    rng = np.random.default_rng(1)
    Xc, yc = hetero_regression(2000, rng)
    Xt, yt = hetero_regression(20000, rng)
    rows = coverage_calibration_curve(mean_true, Xc, yc, Xt, yt)
    assert len(rows) == 5
    for nominal, emp, gap in rows:
        assert gap > -0.02 and gap < 0.02
        assert abs(gap - (emp - nominal)) < 1e-12


def test_coverage_monotone_in_alpha():
    rng = np.random.default_rng(2)
    Xc, yc = hetero_regression(500, rng)
    Xt, yt = hetero_regression(5000, rng)
    emps = [r[1] for r in coverage_calibration_curve(mean_true, Xc, yc, Xt, yt, alphas=(0.5, 0.3, 0.1, 0.05))]
    assert emps == sorted(emps)
