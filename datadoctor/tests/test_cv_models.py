import numpy as np
import pytest

from datadoctor.cv import cross_val_probs, stratified_kfold
from datadoctor.models import KNNClassifier, SoftmaxRegression
from datadoctor.synth import make_blobs


def test_folds_partition_and_are_stratified():
    y = np.array([0] * 50 + [1] * 30 + [2] * 7)
    folds = stratified_kfold(y, 5, seed=1)
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen.tolist()) == list(range(len(y)))
    for tr, te in folds:
        assert not set(tr) & set(te)
        counts = np.bincount(y[te], minlength=3)
        assert abs(counts[0] - 10) <= 1 and abs(counts[1] - 6) <= 1 and counts[2] in (1, 2)


def test_folds_are_deterministic_and_seed_dependent():
    y = np.arange(60) % 3
    a = stratified_kfold(y, 4, 0)
    b = stratified_kfold(y, 4, 0)
    c = stratified_kfold(y, 4, 1)
    assert all((x[1] == z[1]).all() for x, z in zip(a, b))
    assert any(len(x[1]) != len(z[1]) or (x[1] != z[1]).any() for x, z in zip(a, c))


def test_fold_validation():
    with pytest.raises(ValueError):
        stratified_kfold(np.array([0, 1, 0]), 1)
    with pytest.raises(ValueError):
        stratified_kfold(np.array([0, 1]), 5)


def test_softmax_learns_separable_data_and_probs_sum_to_one():
    X, y = make_blobs(n=600, sep=3.0, seed=0)
    m = SoftmaxRegression().fit(X, y)
    P = m.predict_proba(X)
    assert P.sum(axis=1) == pytest.approx(np.ones(len(X)))
    assert (m.predict(X) == y).mean() > 0.97


def test_softmax_unseen_class_gets_tiny_probability():
    X, y = make_blobs(n=300, n_classes=3, sep=3.0, seed=0)
    keep = y != 2
    m = SoftmaxRegression().fit(X[keep], y[keep], n_classes=3)
    assert m.predict_proba(X).shape[1] == 3
    assert m.predict_proba(X)[:, 2].max() < 1e-3


def test_cross_val_probs_are_out_of_sample():
    # A model that memorises (1-NN with k=1) would give P=1 on the own label in-sample;
    # out-of-sample, a *mislabelled* point must get a low probability for its wrong label.
    X, y = make_blobs(n=500, n_classes=3, sep=4.0, seed=2)
    y2 = y.copy()
    y2[0] = (y[0] + 1) % 3
    P = cross_val_probs(X, y2, lambda: KNNClassifier(k=1), k=5)
    assert P[0, y2[0]] < 0.4
    assert P[0, y[0]] >= 0.6 - 1e-9


def test_cross_val_probs_shape_and_normalisation():
    X, y = make_blobs(n=200, n_classes=4, seed=0)
    P = cross_val_probs(X, y, k=4, repeats=2)
    assert P.shape == (200, 4)
    assert P.sum(axis=1) == pytest.approx(np.ones(200))


def test_knn_probabilities_valid():
    X, y = make_blobs(n=200, n_classes=3, sep=3, seed=0)
    P = KNNClassifier(k=7).fit(X, y).predict_proba(X)
    assert P.sum(axis=1) == pytest.approx(np.ones(200))
    assert (P > 0).all()
