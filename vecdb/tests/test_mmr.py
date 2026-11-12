import numpy as np
import pytest

from vecdb import Collection


def make():
    c = Collection(2, "cosine", "flat")
    # three near-duplicates at angle ~0, one distinct at 60 degrees, one at 90
    vecs = [[1, 0], [1, 0.01], [1, 0.02], [0.5, 0.866], [0, 1]]
    c.upsert(["a", "b", "c", "d", "e"], np.array(vecs, dtype=np.float32),
             [{"g": 1}, {"g": 1}, {"g": 1}, {"g": 2}, {"g": 2}])
    return c


def test_lambda_one_is_plain_topk():
    c = make()
    q = [1, 0]
    assert [h.id for h in c.query_mmr(q, k=3, lambda_mult=1.0)] == [h.id for h in c.query(np.array([q], dtype=np.float32), k=3)[0]]


def test_diversity_skips_duplicates():
    ids = [h.id for h in make().query_mmr([1, 0], k=2, lambda_mult=0.3)]
    assert ids[0] == "a" and ids[1] in ("d", "e")


def test_filter_and_bounds():
    c = make()
    assert {h.id for h in c.query_mmr([1, 0], k=5, where={"g": 2})} == {"d", "e"}
    with pytest.raises(ValueError):
        c.query_mmr([1, 0], lambda_mult=1.5)
