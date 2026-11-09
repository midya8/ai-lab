import numpy as np
import pytest

from vecdb import BM25, HybridCollection
from vecdb.hybrid import tokenize


def make():
    rng = np.random.default_rng(0)
    h = HybridCollection(8)
    texts = ["the quick brown fox", "lazy dogs sleep all day", "error code E1234 in module auth",
             "foxes and dogs are mammals", "auth module refactor notes"]
    V = rng.normal(size=(5, 8)).astype(np.float32)
    h.upsert([f"d{i}" for i in range(5)], V, [{"g": i % 2} for i in range(5)], texts)
    return h, V


def test_bm25_rare_term_beats_common():
    bm = BM25([tokenize(t) for t in ["a a a b", "a c", "d e"]])
    s = bm.scores(["c"])
    assert s[1] > 0 and s[0] == 0 and s[2] == 0
    assert bm.idf("c") > bm.idf("a")


def test_bm25_length_normalisation():
    bm = BM25([["x"], ["x"] + ["pad"] * 30])
    s = bm.scores(["x"])
    assert s[0] > s[1]


def test_lexical_finds_exact_token_vectors_miss():
    h, V = make()
    top = h.lexical("E1234", 3)
    assert top[0][0] == "d2" and len(top) == 1


def test_rrf_surfaces_lexical_match_for_unrelated_vector():
    h, V = make()
    q = -V[0]  # vector far from d0; text points at d2
    hits = h.query(q, "E1234 auth", k=3)
    assert "d2" in [x.id for x in hits[:2]]


def test_weighted_alpha_extremes():
    h, V = make()
    dense_only = h.query(V[1], "E1234", k=1, fusion="weighted", alpha=1.0)
    lex_only = h.query(V[1], "E1234", k=1, fusion="weighted", alpha=0.0)
    assert dense_only[0].id == "d1" and lex_only[0].id == "d2"


def test_filters_apply_to_both_retrievers():
    h, V = make()
    for fusion in ("rrf", "weighted"):
        hits = h.query(V[2], "E1234 auth module", k=5, where={"g": 1}, fusion=fusion)
        assert hits and all(x.metadata["g"] == 1 for x in hits)


def test_delete_and_upsert_update_lexical_side():
    h, V = make()
    h.delete("d2")
    assert h.lexical("E1234") == []
    h.upsert("d0", V[0], [{"g": 0}], ["now mentions E1234"])
    assert h.lexical("E1234")[0][0] == "d0"
    h.upsert("d0", V[0], [{"g": 0}])  # no text -> drops lexical entry
    assert h.lexical("E1234") == []


def test_validation():
    h, V = make()
    with pytest.raises(ValueError):
        h.query(V[0], "x", fusion="nope")
    with pytest.raises(ValueError):
        h.query(V[0], "x", fusion="weighted", alpha=2)
    with pytest.raises(ValueError):
        h.upsert("z", V[0], texts=["a", "b"])
