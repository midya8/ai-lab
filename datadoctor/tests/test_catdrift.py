import numpy as np
import pytest

from datadoctor.catdrift import _chi2_sf, categorical_drift


def test_chi2_sf_known_values():
    assert _chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    assert _chi2_sf(9.487729036781154, 4) == pytest.approx(0.05, abs=1e-6)
    assert _chi2_sf(30.0, 5) == pytest.approx(1.4749e-5, rel=1e-3)


def test_no_drift_same_distribution():
    rng = np.random.default_rng(0)
    a = rng.choice(list("abc"), 2000, p=[.5, .3, .2])
    b = rng.choice(list("abc"), 2000, p=[.5, .3, .2])
    r = categorical_drift(a, b)
    assert r["p_value"] > 0.01 and r["psi"] < 0.05 and not r["flagged"]


def test_detects_shift():
    rng = np.random.default_rng(1)
    a = rng.choice(list("abc"), 2000, p=[.5, .3, .2])
    b = rng.choice(list("abc"), 2000, p=[.2, .3, .5])
    r = categorical_drift(a, b)
    assert r["p_value"] < 1e-6 and r["flagged"]


def test_unseen_category_flagged():
    r = categorical_drift(["a"] * 50 + ["b"] * 50, ["a"] * 45 + ["b"] * 45 + ["z"] * 10)
    assert r["unseen_categories"] == ["z"] and r["unseen_rate"] == pytest.approx(0.1) and r["flagged"]


def test_empty_raises():
    with pytest.raises(ValueError):
        categorical_drift([], ["a"])
