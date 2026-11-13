import numpy as np
import pytest

from datadoctor import adversarial_validation, auc


def test_auc_known_values():
    assert auc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0
    assert auc(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1])) == 0.0
    assert auc(np.array([0, 1, 0, 1]), np.array([.5, .5, .5, .5])) == 0.5
    assert auc(np.array([0, 0, 1]), np.array([.1, .5, .5])) == pytest.approx(0.75)


def test_no_shift_not_flagged():
    rng = np.random.default_rng(0)
    r = adversarial_validation(rng.normal(size=(400, 4)), rng.normal(size=(400, 4)), n_permutations=20)
    assert r["auc"] < 0.6 and not r["shifted"]


def test_mean_shift_detected_and_attributed():
    rng = np.random.default_rng(1)
    te = rng.normal(size=(400, 4)); te[:, 2] += 1.0
    r = adversarial_validation(rng.normal(size=(400, 4)), te, n_permutations=20)
    assert r["shifted"] and r["auc"] > 0.7 and r["top_features"][0] == 2


def test_correlation_shift_needs_quadratic_features():
    rng = np.random.default_rng(2)
    n = 800
    a = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 1))
    b = np.hstack([z, z + 0.15 * rng.normal(size=(n, 1))])  # same marginals, strong correlation
    lin = adversarial_validation(a, b, n_permutations=20)
    quad = adversarial_validation(a, b, quadratic=True, n_permutations=20)
    assert lin["auc"] < 0.6
    assert quad["auc"] > 0.8 and quad["shifted"]


def test_shape_mismatch():
    with pytest.raises(ValueError):
        adversarial_validation(np.zeros((5, 2)), np.zeros((5, 3)))
