import numpy as np
from conformal_pred import *


def test_rank_scores_and_topk_sets_hand():
    P = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]])
    assert list(rank_scores(P, [2, 0])) == [1.0, 0.0]
    S = topk_sets(P, 2)
    assert S.tolist() == [[False, True, True], [True, True, False]]


def test_calibrate_topk_covers_and_is_not_larger_than_needed():
    rng = np.random.default_rng(3)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(6000, rng)
    m = SoftmaxRegressor(6).fit(Xtr, ytr)
    k = calibrate_topk(m.predict_proba(Xc), yc, 0.1)
    S = topk_sets(m.predict_proba(Xt), k)
    assert set_coverage(S, yt) >= 0.9 - 0.02
    if k > 1:
        assert set_coverage(topk_sets(m.predict_proba(Xt), k - 1), yt) < 0.9 + 0.01
    # adaptive LAC sets are no larger on average at matching coverage
    cp = ConformalClassifier(m.predict_proba, 0.1).calibrate(Xc, yc)
    assert mean_set_size(cp.predict_sets(Xt)) <= mean_set_size(S) + 0.05


def test_tiny_calibration_falls_back_to_all_labels():
    P = np.full((3, 4), 0.25)
    assert calibrate_topk(P, [0, 1, 2], 0.05) == 4
