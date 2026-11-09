import numpy as np
import pytest

from datadoctor.multilabel import (
    cross_val_probs_multilabel, find_multilabel_issues, inject_multilabel_noise,
    make_multilabel_blobs, rank_multilabel_issues, score_multilabel,
)


@pytest.fixture(scope="module")
def noisy():
    X, Y = make_multilabel_blobs(n=1200, n_labels=5, seed=1)
    Yn, flip = inject_multilabel_noise(Y, 0.05, seed=2)
    P = cross_val_probs_multilabel(X, Yn, seed=3)
    return X, Y, Yn, flip, P


def test_noise_rate_and_shape():
    Y = np.random.default_rng(0).integers(0, 2, (100, 4))
    Yn, flip = inject_multilabel_noise(Y, 0.1)
    assert flip.sum() == 40 and (Yn != Y).sum() == 40 and Yn.shape == Y.shape


def test_probs_in_range(noisy):
    P = noisy[4]
    assert P.shape == noisy[2].shape and P.min() >= 0 and P.max() <= 1


def test_detects_noise_better_than_chance(noisy):
    _, _, Yn, flip, P = noisy
    s = score_multilabel(find_multilabel_issues(P, Yn), flip)
    assert s["precision"] > 3 * flip.mean() and s["recall"] > 0.3


def test_clean_labels_flag_little():
    X, Y = make_multilabel_blobs(n=1000, n_labels=4, seed=5)
    P = cross_val_probs_multilabel(X, Y, seed=1)
    assert find_multilabel_issues(P, Y).mean() < 0.06


def test_ranking_kinds_and_order(noisy):
    _, _, Yn, _, P = noisy
    mask = find_multilabel_issues(P, Yn)
    cells = rank_multilabel_issues(P, Yn, mask)
    assert len(cells) == mask.sum()
    assert [c[3] for c in cells] == sorted(c[3] for c in cells)
    for i, k, kind, _ in cells[:50]:
        assert kind == ("missing" if Yn[i, k] == 0 else "spurious")


def test_constant_label_skipped():
    P = np.random.default_rng(0).uniform(size=(50, 2))
    Y = np.stack([np.zeros(50, int), np.arange(50) % 2], axis=1)
    assert not find_multilabel_issues(P, Y)[:, 0].any()


def test_validation():
    with pytest.raises(ValueError):
        find_multilabel_issues(np.zeros((3, 2)), np.zeros((3, 3), int))
    with pytest.raises(ValueError):
        find_multilabel_issues(np.zeros((3, 2)), np.full((3, 2), 2))
