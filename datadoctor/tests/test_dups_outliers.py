import numpy as np
import pytest

from datadoctor.dups import (
    conflicting_groups,
    exact_duplicate_groups,
    jaccard,
    leakage_pairs,
    near_duplicate_pairs,
    near_duplicate_texts,
    shingles,
)
from datadoctor.outliers import find_outliers, robust_z
from datadoctor.synth import add_duplicates, add_outliers, make_blobs


def test_exact_duplicates_and_conflicts():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [1.0, 2.0], [5.0, 6.0], [3.0, 4.0]])
    y = np.array([0, 1, 0, 1, 2, 1])
    g = sorted(sorted(x) for x in exact_duplicate_groups(X))
    assert g == [[0, 2, 3], [1, 5]]
    assert [sorted(c) for c in conflicting_groups(g, y)] == [[0, 2, 3]]


def test_near_duplicates_found_with_jitter_and_not_otherwise():
    X, y = make_blobs(n=400, seed=0)
    X2, y2, src = add_duplicates(X, y, frac=0.05, jitter=1e-3, seed=0)
    pairs = near_duplicate_pairs(X2)
    found = {(a, b) for a, b, _ in pairs}
    for k, s in enumerate(src):
        assert (int(s), 400 + k) in found
    assert len(pairs) <= len(src) + 2  # essentially nothing spurious
    assert near_duplicate_pairs(X) == [] or len(near_duplicate_pairs(X)) <= 2


def test_leakage_pairs_standardised_with_train_stats():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(200, 5))
    test = rng.normal(size=(50, 5))
    test[7] = train[33] + 1e-4
    out = leakage_pairs(train, test)
    assert (33, 7) in [(tr, te) for tr, te, _ in out]  # (train_idx, test_idx)
    assert len(out) <= 2


def test_text_shingles_and_jaccard():
    assert shingles("a b", 3) == {"a b"}
    assert shingles("", 3) == set()
    assert jaccard({"a"}, {"a"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_minhash_lsh_finds_near_dupes_without_false_positives():
    base = "the quick brown fox jumps over the lazy dog while the cat sleeps quietly on the warm windowsill today"
    near = base.replace("today", "tonight")
    other = "completely unrelated sentence about training a transformer model on a large corpus of text data"
    texts = [base, other, near, "short", other + " again"]
    pairs = near_duplicate_texts(texts, threshold=0.7)
    assert (0, 2) in [(a, b) for a, b, _ in pairs]
    for a, b, j in pairs:
        assert j >= 0.7  # verified exactly
    assert (0, 1) not in [(a, b) for a, b, _ in pairs]


def test_text_threshold_validation():
    with pytest.raises(ValueError):
        near_duplicate_texts(["a b c d", "a b c d"], num_perm=64, bands=10)


def test_robust_z_is_not_dragged_by_outliers():
    s = np.concatenate([np.random.default_rng(0).normal(10, 1, 200), [1000.0]])
    z = robust_z(s)
    assert z[-1] > 100 and abs(np.median(z)) < 0.1


def test_find_outliers_recovers_planted_points():
    X, y = make_blobs(n=600, sep=2.0, seed=0)
    X2, y2, mask = add_outliers(X, y, count=8, scale=8.0, seed=0)
    flagged, z = find_outliers(X2, y2)
    assert flagged[mask].mean() >= 0.9
    assert flagged[~mask].mean() < 0.03
