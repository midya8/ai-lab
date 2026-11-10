import pytest

from tests.test_hybrid import make
from vecdb import HybridCollection


def test_defaults_unchanged():
    h, _ = make()
    assert (h.bm25_k1, h.bm25_b) == (1.5, 0.75)


def test_b_zero_disables_length_normalisation():
    h0 = HybridCollection(2, bm25_b=0.0)
    h1 = HybridCollection(2, bm25_b=1.0)
    for h in (h0, h1):
        h.upsert(["s", "l", "x"], [[1, 0], [0, 1], [1, 1]],
                 texts=["cat", "cat " + "pad " * 30, "other"])
    s0 = dict(h0.lexical("cat"))
    s1 = dict(h1.lexical("cat"))
    assert s0["s"] == pytest.approx(s0["l"])
    assert s1["s"] > s1["l"]


def test_params_persist(tmp_path):
    h = HybridCollection(2, bm25_k1=2.2, bm25_b=0.3)
    h.upsert(["a", "b"], [[1, 0], [0, 1]], texts=["x y", "y z"])
    h.save(str(tmp_path))
    g = HybridCollection.load(str(tmp_path))
    assert (g.bm25_k1, g.bm25_b) == (2.2, 0.3)
    assert g.lexical("x") == h.lexical("x")


def test_validation():
    with pytest.raises(ValueError):
        HybridCollection(2, bm25_b=1.5)
    with pytest.raises(ValueError):
        HybridCollection(2, bm25_k1=-1)
