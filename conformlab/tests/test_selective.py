import numpy as np
from conformlab import *


def test_selective_hand_example():
    S = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], bool)
    y = np.array([0, 0, 1, 0, 1])
    rate, acc = selective_accuracy(S, y)
    assert rate == 0.6 and abs(acc - 2 / 3) < 1e-12


def test_selective_none_answered():
    S = np.ones((4, 3), bool)
    rate, acc = selective_accuracy(S, np.zeros(4, int))
    assert rate == 0.0 and np.isnan(acc)


def test_selective_beats_overall_accuracy():
    rng = np.random.default_rng(3)
    Xtr, ytr = gaussian_classes(3000, rng)
    Xc, yc = gaussian_classes(3000, rng)
    Xt, yt = gaussian_classes(3000, rng)
    m = SoftmaxRegressor(6).fit(Xtr, ytr)
    cp = ConformalClassifier(m.predict_proba, 0.1).calibrate(Xc, yc)
    S = cp.predict_sets(Xt)
    rate, acc = selective_accuracy(S, yt)
    top1 = float((m.predict_proba(Xt).argmax(1) == yt).mean())
    assert 0 < rate < 1 and acc > top1
