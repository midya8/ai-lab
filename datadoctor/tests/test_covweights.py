import numpy as np
import pytest

from datadoctor.covweights import covariate_shift_weights


def test_no_shift_weights_near_uniform():
    rng = np.random.default_rng(0)
    r = covariate_shift_weights(rng.normal(size=(400, 2)), rng.normal(size=(400, 2)))
    assert r["ess_fraction"] > 0.85
    assert r["weights"].mean() == pytest.approx(1.0)


def test_weighting_recovers_test_mean():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(1500, 1))
    B = rng.normal(size=(1500, 1)) + 0.7
    r = covariate_shift_weights(A, B)
    plain = A.mean()
    weighted = float((r["weights"] * A[:, 0]).mean())
    assert abs(weighted - B.mean()) < abs(plain - B.mean()) * 0.5
    assert r["ess_fraction"] < 0.95


def test_clip_and_validation():
    rng = np.random.default_rng(2)
    r = covariate_shift_weights(rng.normal(size=(300, 1)), rng.normal(size=(300, 1)) + 3, clip=3.0)
    assert r["ess_fraction"] > 0.0 and np.isfinite(r["weights"]).all()
    with pytest.raises(ValueError):
        covariate_shift_weights(np.zeros((5, 2)), np.zeros((5, 3)))
