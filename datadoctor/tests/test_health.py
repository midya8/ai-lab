import numpy as np
import pytest

from datadoctor import feature_health


def test_detects_all_issues():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = np.column_stack([a, 2 * a + 1, rng.normal(size=200), np.full(200, 3.0)])
    X[:20, 2] = np.nan
    h = feature_health(X, names=["a", "a2", "n", "c"])
    assert h["constant"] == ["c"]
    assert h["missing"] == {"n": pytest.approx(0.1)}
    assert len(h["redundant_pairs"]) == 1
    p = h["redundant_pairs"][0]
    assert (p["a"], p["b"]) == ("a", "a2") and p["r"] == pytest.approx(1.0)


def test_negative_correlation_and_clean_data():
    rng = np.random.default_rng(1)
    a = rng.normal(size=100)
    h = feature_health(np.column_stack([a, -a]))
    assert h["redundant_pairs"][0]["r"] == pytest.approx(-1.0)
    clean = feature_health(rng.normal(size=(100, 3)))
    assert clean == {"missing": {}, "constant": [], "redundant_pairs": []}


def test_all_nan_column_is_constant_and_bad_shape():
    X = np.column_stack([np.arange(5.0), np.full(5, np.nan)])
    assert feature_health(X)["constant"] == ["f1"]
    with pytest.raises(ValueError):
        feature_health(np.arange(5.0))
