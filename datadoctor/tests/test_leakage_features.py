import numpy as np
import pytest

from datadoctor import leaky_features


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    leak = y * 10.0 + rng.normal(0, 0.5, n)       # near-perfect proxy
    weak = y * 0.3 + rng.normal(0, 1.0, n)        # mildly informative
    noise = rng.normal(size=n)
    const = np.full(n, 2.0)
    return np.column_stack([leak, weak, noise, const]), y


def test_flags_proxy_only():
    X, y = _data()
    r = leaky_features(X, y, names=["leak", "weak", "noise", "const"])
    assert [x["feature"] for x in r][0] == "leak"
    assert r[0]["flagged"] and r[0]["accuracy"] > 0.95
    assert not any(x["flagged"] for x in r[1:])
    assert "const" not in [x["feature"] for x in r]


def test_noise_near_baseline():
    X, y = _data()
    r = {x["feature"]: x for x in leaky_features(X, y, names=["leak", "weak", "noise", "const"])}
    assert abs(r["noise"]["accuracy"] - r["noise"]["baseline"]) < 0.1


def test_nan_handling_and_validation():
    X, y = _data()
    X[:100, 0] = np.nan
    assert leaky_features(X, y)[0]["feature"] == "f0"
    with pytest.raises(ValueError):
        leaky_features(X, y[:-1])
