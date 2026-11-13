import numpy as np
import pytest
from datadoctor.wasserstein import wasserstein1, wasserstein_drift


def test_pure_shift_equals_shift():
    x = np.random.default_rng(0).normal(size=500)
    assert wasserstein1(x, x + 0.7) == pytest.approx(0.7, abs=1e-9)


def test_identical_is_zero_and_symmetric():
    r = np.random.default_rng(1)
    a, b = r.normal(size=200), r.normal(1, 2, size=300)
    assert wasserstein1(a, a) == 0.0
    assert wasserstein1(a, b) == pytest.approx(wasserstein1(b, a))


def test_matches_scipy_style_quantile_formula_equal_sizes():
    r = np.random.default_rng(2)
    a, b = r.normal(size=100), r.normal(0.5, size=100)
    assert wasserstein1(a, b) == pytest.approx(np.mean(np.abs(np.sort(a) - np.sort(b))))


def test_drift_flags_only_shifted_feature_and_constant_ok():
    r = np.random.default_rng(3)
    tr = np.column_stack([r.normal(size=1000), r.normal(size=1000), np.ones(1000)])
    te = np.column_stack([r.normal(size=1000), r.normal(1.0, size=1000), np.ones(1000)])
    out = wasserstein_drift(tr, te, names=["a", "b", "c"])
    assert out["flagged"] == ["b"] and out["features"][0]["feature"] == "b"
    assert out["features"][0]["w1_norm"] == pytest.approx(1.0, abs=0.1)


def test_bad_inputs():
    with pytest.raises(ValueError):
        wasserstein1([], [1.0])
    with pytest.raises(ValueError):
        wasserstein_drift(np.zeros((3, 2)), np.zeros((3, 3)))
