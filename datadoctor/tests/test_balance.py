import numpy as np
import pytest

from datadoctor import class_balance, label_shift


def test_balance_counts_and_rare():
    y = np.array([0] * 90 + [1] * 8 + [2] * 2)
    b = class_balance(y)
    assert b["counts"] == {0: 90, 1: 8, 2: 2}
    assert b["imbalance_ratio"] == 45.0
    assert b["rare_classes"] == [2]


def test_balance_empty_raises():
    with pytest.raises(ValueError):
        class_balance(np.array([]))


def test_shift_identical_is_zero():
    y = np.array([0, 1, 1, 0, 2, 2])
    s = label_shift(y, y)
    assert s["tv_distance"] == 0 and s["chi2"] == 0 and s["dof"] == 2


def test_shift_detects_change():
    s = label_shift(np.array([0] * 50 + [1] * 50), np.array([0] * 90 + [1] * 10))
    assert s["tv_distance"] == pytest.approx(0.4)
    assert s["chi2"] == pytest.approx(64.0)  # (90-50)^2/50 + (10-50)^2/50 over n=100


def test_shift_unseen_class():
    s = label_shift(np.array([0, 0, 1]), np.array([0, 2]))
    assert s["unseen_in_train"] == [2] and s["chi2"] == float("inf")
