import numpy as np
import pytest

from vecdb import Collection


def make():
    c = Collection(2, "cosine", "flat")
    vecs = [[1, 0], [1, 0.01], [1, 0.02], [0.5, 0.866], [0, 1]]
    c.upsert(["a1", "a2", "a3", "b1", "c1"], np.array(vecs, dtype=np.float32),
             [{"doc": "A"}, {"doc": "A"}, {"doc": "A"}, {"doc": "B"}, {"doc": "C"}])
    return c


def test_one_per_group_in_rank_order():
    out = make().query_grouped([1, 0], "doc", k=3)
    assert [g for g, _ in out] == ["A", "B", "C"]
    assert out[0][1][0].id == "a1" and all(len(h) == 1 for _, h in out)


def test_per_group_and_k_limits():
    out = make().query_grouped([1, 0], "doc", k=2, per_group=2)
    assert [g for g, _ in out] == ["A", "B"]
    assert [h.id for h in out[0][1]] == ["a1", "a2"]


def test_missing_field_skipped_and_where():
    c = make()
    c.upsert(["z"], np.array([[1, 0]], dtype=np.float32), [{}])
    assert "z" not in [h.id for _, hs in c.query_grouped([1, 0], "doc", k=5) for h in hs]
    out = c.query_grouped([1, 0], "doc", k=5, where={"doc": {"$ne": "A"}})
    assert [g for g, _ in out] == ["B", "C"]


def test_validation():
    with pytest.raises(ValueError):
        make().query_grouped([1, 0], "doc", k=0)
