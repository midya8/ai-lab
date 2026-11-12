import numpy as np
import pytest

from datadoctor import feature_drift, ks_statistic, psi


def test_ks_identical_and_disjoint():
    a = np.arange(100.0)
    assert ks_statistic(a, a) == (0.0, 1.0)
    d, p = ks_statistic(a, a + 1000)
    assert d == 1.0 and p < 1e-10


def test_ks_known_value():
    d, _ = ks_statistic(np.array([1.0, 2, 3, 4]), np.array([3.0, 4, 5, 6]))
    assert d == pytest.approx(0.5)


def test_psi_zero_for_same_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.25


def test_feature_drift_finds_shifted_column():
    rng = np.random.default_rng(1)
    tr, te = rng.normal(size=(800, 4)), rng.normal(size=(800, 4))
    te[:, 2] += 1.0
    r = feature_drift(tr, te, names=["a", "b", "c", "d"])
    assert r["flagged"] == ["c"] and r["features"][0]["feature"] == "c"


def test_feature_drift_no_false_alarm_and_shape_check():
    rng = np.random.default_rng(2)
    r = feature_drift(rng.normal(size=(1000, 5)), rng.normal(size=(1000, 5)))
    assert r["flagged"] == []
    with pytest.raises(ValueError):
        feature_drift(np.zeros((5, 2)), np.zeros((5, 3)))
