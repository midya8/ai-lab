import numpy as np
import pytest
from conformal_pred import *


def test_ssc_hand_example():
    S = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 1]], bool)
    y = np.array([0, 0, 1, 0, 2])
    r = size_stratified_coverage(S, y)
    assert r[1] == (0.5, 2) and r[2] == (1.0, 2) and r[3] == (1.0, 1)


def test_worst_stratum_min_count():
    S = np.zeros((40, 3), bool)
    S[:, 0] = True
    y = np.zeros(40, int)
    y[:2] = 1  # size-1 stratum: 95% coverage
    assert worst_size_stratum(S, y, 30) == (1, 0.95)
    with pytest.raises(ValueError):
        worst_size_stratum(S, y, 100)


def test_ssc_on_classifier_counts_sum_and_marginal_consistent():
    rng = np.random.default_rng(0)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(10000, rng)
    m = SoftmaxRegressor(6).fit(Xtr, ytr)
    S = ConformalClassifier(m.predict_proba, 0.1, "lac").calibrate(Xc, yc).predict_sets(Xt)
    r = size_stratified_coverage(S, yt)
    assert sum(c for _, c in r.values()) == len(yt)
    marg = sum(v * c for v, c in r.values()) / len(yt)
    assert abs(marg - set_coverage(S, yt)) < 1e-9
