import numpy as np
import pytest

from test_hybrid import make


def test_must_and_must_not():
    h, V = make()
    q = V[0]
    base = [x.id for x in h.query(q, "auth module", k=5)]
    got = [x.id for x in h.query_terms(q, "auth module", k=5, must=["auth"])]
    assert set(got) == {"d2", "d4"}
    assert got == [i for i in base if i in got]  # relative order preserved
    got = [x.id for x in h.query_terms(q, "auth module", k=5, must_not=["auth", "dogs"])]
    assert set(got) == {"d0"}


def test_no_constraints_equals_query_and_k_respected():
    h, V = make()
    assert [x.id for x in h.query_terms(V[1], "dogs", k=3)] == [x.id for x in h.query(V[1], "dogs", k=3)]
    assert len(h.query_terms(V[1], "dogs", k=1, must_not=["zzz"])) == 1


def test_textless_doc_never_matches_must():
    h, V = make()
    h.upsert("x", V[0])
    assert "x" not in [t.id for t in h.query_terms(V[0], "fox", k=10, must=["fox"])]
    assert h.query_terms(V[0], "fox", k=10, must=["nonexistentword"]) == []
