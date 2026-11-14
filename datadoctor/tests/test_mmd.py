import numpy as np
import pytest

from datadoctor.mmd import mmd_drift, median_bandwidth


def test_same_distribution_not_flagged():
    rng = np.random.default_rng(0)
    r = mmd_drift(rng.normal(size=(150, 3)), rng.normal(size=(150, 3)), n_perm=99, alpha=0.05)
    assert not r["drifted"] and r["p_value"] > 0.05


def test_mean_shift_flagged():
    rng = np.random.default_rng(1)
    r = mmd_drift(rng.normal(size=(150, 3)), rng.normal(size=(150, 3)) + 0.8, n_perm=99, alpha=0.05)
    assert r["drifted"] and r["mmd2"] > 0


def test_correlation_only_shift_caught_where_marginals_match():
    from datadoctor.drift import feature_drift
    rng = np.random.default_rng(2)
    a = rng.normal(size=(400, 2))
    c = np.array([[1, 0.9], [0.9, 1]])
    b = rng.normal(size=(400, 2)) @ np.linalg.cholesky(c).T
    assert not feature_drift(a, b)["flagged"]
    assert mmd_drift(a, b, n_perm=99, seed=3, alpha=0.05)["drifted"]


def test_validation_and_bandwidth():
    with pytest.raises(ValueError):
        mmd_drift(np.zeros((5, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        mmd_drift(np.zeros((1, 2)), np.zeros((5, 2)))
    assert median_bandwidth(np.zeros((4, 2))) == 1.0
