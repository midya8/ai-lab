import numpy as np
import pytest

from vecdb import Collection


def make(index="flat"):
    c = Collection(2, "cosine", index)
    vecs = [[1, 0], [1, 0.01], [0, 1], [0.01, 1], [1, 1]]
    c.upsert(list("abcde"), np.array(vecs, dtype=np.float32))
    return c


def test_finds_pairs_once_sorted():
    d = make().find_duplicates(0.99)
    assert {(a, b) for a, b, _ in d} == {("a", "b"), ("c", "d")}
    assert all(c >= 0.99 for *_, c in d)
    assert d[0][2] >= d[1][2]


def test_ignores_deleted_and_small():
    c = make()
    c.delete(["b"])
    assert {(a, b) for a, b, _ in c.find_duplicates(0.99)} == {("c", "d")}
    assert Collection(2, "cosine", "flat").find_duplicates() == []


def test_bad_threshold_and_hnsw():
    with pytest.raises(ValueError):
        make().find_duplicates(2)
    assert ("a", "b") in {(a, b) for a, b, _ in make("hnsw").find_duplicates(0.99)}
