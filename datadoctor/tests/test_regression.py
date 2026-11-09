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


def test_ordinal_noise_found_and_suggestions_sane():
    from datadoctor.regression import find_ordinal_issues, inject_ordinal_noise
    rng = np.random.default_rng(3)
    X = rng.normal(size=(800, 5))
    latent = X @ np.array([1.5, -1.0, 0.8, 0.0, 0.5]) + rng.normal(size=800) * 0.3
    y = np.digitize(latent, np.quantile(latent, [0.2, 0.4, 0.6, 0.8])) + 1.0  # levels 1..5
    yn, mask = inject_ordinal_noise(y, 0.08, seed=2)
    assert mask.sum() > 0 and np.all(np.abs(yn[mask] - y[mask]) >= 2)
    issues, gaps, sugg = find_ordinal_issues(X, yn, min_gap=1.5)
    assert mask[issues].mean() > 0.7 and mask[issues].sum() / mask.sum() > 0.6
    assert np.all(gaps[issues[:-1]] >= gaps[issues[1:]])
    assert sugg.min() >= 1 and sugg.max() <= 5
    # suggested level matches the true one for most caught rows
    caught = issues[mask[issues]]
    assert (sugg[caught] == y[caught]).mean() > 0.5


def test_ordinal_clean_data_flags_little():
    from datadoctor.regression import find_ordinal_issues
    rng = np.random.default_rng(4)
    X = rng.normal(size=(600, 4))
    latent = X @ np.array([1.0, 1.0, -1.0, 0.5])
    y = np.digitize(latent, np.quantile(latent, [0.25, 0.5, 0.75])) + 1.0
    issues, _, _ = find_ordinal_issues(X, y, min_gap=1.5)
    assert len(issues) < 0.03 * len(y)
