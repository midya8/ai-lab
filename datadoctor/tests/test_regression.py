import numpy as np
import pytest

from datadoctor.regression import (RidgeRegression, cross_val_predict, find_target_issues,
                                   inject_target_noise, robust_residual_scores)


def _data(n=600, seed=0, hetero=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    w = np.array([2.0, -1.0, 0.5, 0.0, 1.5, -2.0])
    noise = rng.normal(size=n) * (0.3 + (0.6 * np.abs(X[:, 0]) if hetero else 0))
    return X, X @ w + 1.0 + noise


def test_ridge_recovers_linear_model():
    X, y = _data()
    m = RidgeRegression(l2=1e-3).fit(X, y)
    assert np.corrcoef(m.predict(X), y)[0, 1] > 0.98


def test_cross_val_predict_is_out_of_fold():
    X, y = _data(n=100)
    # a model that memorises its training rows would be exact in-sample; OOF must not be
    pred = cross_val_predict(X, y, lambda: RidgeRegression(1.0), k=5)
    assert np.abs(pred - y).mean() > 0.05
    with pytest.raises(ValueError):
        cross_val_predict(X, y, k=1)


def test_injected_noise_is_found():
    X, y = _data()
    yn, mask = inject_target_noise(y, 0.08, seed=1)
    assert mask.sum() == round(0.08 * len(y))
    issues, scores, _ = find_target_issues(X, yn)
    hit = mask[issues].mean()
    recall = mask[issues].sum() / mask.sum()
    assert hit > 0.8 and recall > 0.7


def test_clean_data_flags_little():
    X, y = _data()
    issues, _, _ = find_target_issues(X, y)
    assert len(issues) / len(y) < 0.03


def test_local_mad_protects_noisy_region():
    X, y = _data(n=1500, hetero=True, seed=2)
    pred = cross_val_predict(X, y)
    local = robust_residual_scores(pred, y, bins=5)
    # with one global bin, the high-variance rows get flagged far more often
    # (|x0| drives variance but not the prediction, so compare by |x0|)
    glob = robust_residual_scores(pred, y, bins=1)
    hi = np.abs(X[:, 0]) > 1.5
    assert (glob[hi] > 3.5).mean() >= (local[hi] > 3.5).mean()


def test_ranked_descending():
    X, y = _data()
    yn, _ = inject_target_noise(y, 0.1, seed=3)
    issues, scores, _ = find_target_issues(X, yn)
    s = scores[issues]
    assert (np.diff(s) <= 1e-12).all()
