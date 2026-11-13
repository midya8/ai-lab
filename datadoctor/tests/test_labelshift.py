import numpy as np
import pytest

from datadoctor.labelshift import bbse_weights


def _noisy(y, rng, acc=0.8, k=3):
    p = y.copy()
    flip = rng.random(y.size) > acc
    p[flip] = (y[flip] + rng.integers(1, k, flip.sum())) % k
    return p


def test_recovers_prior_shift():
    rng = np.random.default_rng(0)
    yv = rng.integers(0, 3, 6000)
    yt = rng.choice(3, 6000, p=[0.7, 0.2, 0.1])
    r = bbse_weights(yv, _noisy(yv, rng), _noisy(yt, rng))
    assert np.allclose(r["test_prior"], [0.7, 0.2, 0.1], atol=0.04)


def test_no_shift_weights_near_one():
    rng = np.random.default_rng(1)
    yv = rng.integers(0, 3, 6000)
    yt = rng.integers(0, 3, 6000)
    r = bbse_weights(yv, _noisy(yv, rng), _noisy(yt, rng))
    assert np.allclose(r["weights"], 1.0, atol=0.12)


def test_weights_nonnegative_and_prior_sums_to_one():
    rng = np.random.default_rng(2)
    yv = rng.integers(0, 3, 500)
    r = bbse_weights(yv, _noisy(yv, rng), _noisy(yv, rng))
    assert (r["weights"] >= 0).all() and abs(r["test_prior"].sum() - 1) < 1e-9


def test_singular_raises():
    yv = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError):
        bbse_weights(yv, np.zeros(4, int), np.array([0, 1]))


def test_shape_mismatch():
    with pytest.raises(ValueError):
        bbse_weights([0, 1], [0], [0])
