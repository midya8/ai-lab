import numpy as np
import pytest
from conformal_pred import *


def test_outcome_hand_example():
    S = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1]], bool)
    y = np.array([0, 0, 1, 0, 0])
    r = set_outcome_breakdown(S, y)
    assert r == {"empty": 0.2, "singleton_correct": 0.2, "singleton_wrong": 0.2,
                 "multi_covered": 0.2, "multi_missed": 0.2}


def test_outcome_sums_to_one_and_matches_coverage():
    rng = np.random.default_rng(1)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(3000, rng)
    m = SoftmaxRegressor(6).fit(Xtr, ytr)
    cp = ConformalClassifier(m.predict_proba, 0.1).calibrate(Xc, yc)
    S = cp.predict_sets(Xt)
    r = set_outcome_breakdown(S, yt)
    assert abs(sum(r.values()) - 1) < 1e-12
    cov = r["singleton_correct"] + r["multi_covered"]
    assert abs(cov - set_coverage(S, yt)) < 1e-12


def test_worst_class_coverage():
    S = np.zeros((90, 3), bool)
    y = np.repeat([0, 1, 2], 30)
    S[y == 0, 0] = True
    S[y == 1, 1] = True
    S[y == 2, 2] = True
    S[:10, 0] = False
    assert worst_class_coverage(S, y, 3) == (0, pytest.approx(2 / 3))
    with pytest.raises(ValueError):
        worst_class_coverage(S, y, 3, min_count=100)
