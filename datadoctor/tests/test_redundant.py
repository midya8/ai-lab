import numpy as np
import pytest
from datadoctor.redundant import drop_redundant, redundant_features


def data():
    r = np.random.default_rng(0)
    a = r.normal(size=200)
    b = r.normal(size=200)
    return np.c_[a, b, 2 * a + 1, -b, np.ones(200), a.copy()]


def test_finds_pairs_and_signs():
    p = redundant_features(data(), list("abcdef"))
    pairs = {(q["a"], q["b"]): q for q in p}
    assert set(pairs) == {("a", "c"), ("a", "f"), ("b", "d"), ("c", "f")}
    assert pairs[("b", "d")]["r"] == pytest.approx(-1)
    assert pairs[("a", "f")]["exact"] and not pairs[("a", "c")]["exact"]


def test_constant_skipped_and_independent_clean():
    r = np.random.default_rng(1)
    assert redundant_features(np.c_[r.normal(size=(100, 3)), np.ones(100)]) == []


def test_drop_keeps_first_of_group():
    assert drop_redundant(data(), list("abcdef")) == [0, 1, 4]


def test_validation():
    with pytest.raises(ValueError):
        redundant_features(np.zeros(3))
    with pytest.raises(ValueError):
        redundant_features(np.zeros((3, 2)), ["a"])
    assert redundant_features(np.zeros((1, 2))) == []
