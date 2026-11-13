
import numpy as np
import pytest

from datadoctor import informative_missingness
from datadoctor.missingness import _chi2_sf


def test_chi2_sf_known_values():
    assert _chi2_sf(3.841458820694124, 1) == pytest.approx(0.05, abs=1e-6)
    assert _chi2_sf(5.991464547107979, 2) == pytest.approx(0.05, abs=1e-6)
    assert _chi2_sf(30.0, 4) == pytest.approx(4.8e-6, rel=0.05)
    assert _chi2_sf(0.0, 3) == 1.0


def test_flags_label_dependent_missingness_only():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, 3))
    X[(y == 1) & (rng.random(n) < 0.6), 0] = np.nan   # informative
    X[rng.random(n) < 0.3, 1] = np.nan                # MCAR
    hits = informative_missingness(X, y, names=["a", "b", "c"])
    assert [h["feature"] for h in hits] == ["a"]
    assert hits[0]["missing_rate_by_class"]["0"] == 0.0
    assert hits[0]["missing_rate_by_class"]["1"] > 0.4


def test_skips_rare_and_validates():
    X = np.ones((50, 1)); X[0, 0] = np.nan
    assert informative_missingness(X, np.zeros(50, int)) == []
    with pytest.raises(ValueError):
        informative_missingness(X, np.zeros(3))
