import numpy as np
import pytest

from vecdb import HybridCollection


def make():
    h = HybridCollection(2)
    vecs = np.array([[1, 0], [1, 0.01], [0.5, 0.866], [0, 1]], dtype=np.float32)
    h.upsert(["a1", "a2", "b1", "c1"], vecs,
             [{"doc": "A"}, {"doc": "A"}, {"doc": "B"}, {"doc": "C"}],
             ["red apple pie", "apple tart recipe", "banana bread", "cherry cake"])
    return h


def test_groups_collapse_and_order():
    out = make().query_grouped([1, 0], "apple", "doc", k=3)
    assert [g for g, _ in out][0] == "A"
    assert all(len(hs) == 1 for _, hs in out)
    assert len({g for g, _ in out}) == len(out)


def test_per_group_and_kwargs():
    h = make()
    out = h.query_grouped([1, 0], "apple", "doc", k=1, per_group=2)
    assert [x.id for x in out[0][1]] == ["a1", "a2"] or sorted(x.id for x in out[0][1]) == ["a1", "a2"]
    out = h.query_grouped([1, 0], "apple", "doc", k=5, where={"doc": {"$ne": "A"}})
    assert "A" not in [g for g, _ in out]


def test_missing_field_and_validation():
    h = make()
    h.upsert(["z"], np.array([[1, 0]], dtype=np.float32), [{}], ["apple"])
    ids = [x.id for _, hs in h.query_grouped([1, 0], "apple", "doc", k=9) for x in hs]
    assert "z" not in ids
    with pytest.raises(ValueError):
        h.query_grouped([1, 0], "apple", "doc", k=0)
