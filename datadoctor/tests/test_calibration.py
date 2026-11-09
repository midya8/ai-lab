import numpy as np
import pytest

from datadoctor.calibration import (
    apply_temperature,
    calibrate,
    expected_calibration_error,
    fit_temperature,
    nll,
)
from datadoctor.confident import find_label_issues


def _noisy_probs(n=3000, K=4, T_true=3.0, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, K)) * 1.5
    p_true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(K, p=p) for p in p_true])
    # the model reports an over-confident version of the real distribution
    P = apply_temperature(p_true, 1.0 / T_true)
    return P, y


def test_temperature_one_is_identity():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert np.allclose(apply_temperature(P, 1.0), P)


def test_rows_sum_to_one_and_order_preserved():
    P = np.array([[0.6, 0.3, 0.1]])
    Q = apply_temperature(P, 2.5)
    assert Q.sum() == pytest.approx(1.0)
    assert list(np.argsort(Q[0])) == list(np.argsort(P[0]))
    assert Q.max() < P.max()


def test_bad_temperature():
    with pytest.raises(ValueError):
        apply_temperature(np.array([[0.5, 0.5]]), 0.0)


def test_fit_recovers_overconfidence_and_lowers_nll_and_ece():
    P, y = _noisy_probs()
    Pc, T = calibrate(P, y)
    assert 2.0 < T < 4.5  # true value 3
    assert nll(Pc, y) < nll(P, y)
    assert expected_calibration_error(Pc, y) < expected_calibration_error(P, y)


def test_well_calibrated_gives_t_near_one():
    P, y = _noisy_probs(T_true=1.0, seed=3)
    assert fit_temperature(P, y) == pytest.approx(1.0, abs=0.2)


def test_find_label_issues_still_runs_on_calibrated():
    P, y = _noisy_probs(n=600)
    Pc, _ = calibrate(P, y)
    assert find_label_issues(Pc, y).shape == (600,)
