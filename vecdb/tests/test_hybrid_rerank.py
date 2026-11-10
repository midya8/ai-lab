import pytest

from tests.test_hybrid import make


def overlap(q, docs):
    qs = set(q.split())
    return [len(qs & set(d.split())) for d in docs]


def test_scorer_alone_decides():
    h, V = make()
    hits = h.query_rerank(V[0], "auth module", overlap, k=2, candidates=5)
    assert hits[0].id in {"d2", "d4"}


def test_blend_zero_matches_first_stage_order():
    h, V = make()
    base = [x.id for x in h.query(V[1], "dogs", k=3)]
    out = [x.id for x in h.query_rerank(V[1], "dogs", overlap, k=3, candidates=5, blend=0.0)]
    assert out == base


def test_filter_passthrough_and_k():
    h, V = make()
    hits = h.query_rerank(V[2], "auth", overlap, k=2, where={"g": 0})
    assert len(hits) <= 2 and all(x.metadata["g"] == 0 for x in hits)


def test_validation():
    h, V = make()
    with pytest.raises(ValueError):
        h.query_rerank(V[0], "x", overlap, blend=2)
    with pytest.raises(ValueError):
        h.query_rerank(V[0], "fox", lambda q, d: [1.0], k=3)
