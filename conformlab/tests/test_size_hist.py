import numpy as np
from conformlab import *


def test_hist_hand_example():
    S = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1]], bool)
    h = set_size_histogram(S)
    assert np.allclose(h, [0.25, 0.25, 0.25, 0.25])


def test_hist_sums_to_one_and_matches_mean():
    rng = np.random.default_rng(0)
    S = rng.random((200, 5)) < 0.4
    h = set_size_histogram(S)
    assert len(h) == 6 and abs(h.sum() - 1) < 1e-12
    assert abs((h * np.arange(6)).sum() - mean_set_size(S)) < 1e-12


def test_hist_full_sets():
    assert set_size_histogram(np.ones((3, 6), bool))[6] == 1.0
