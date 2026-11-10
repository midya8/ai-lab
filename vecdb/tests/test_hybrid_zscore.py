import numpy as np
import pytest

from tests.test_hybrid import make


def test_zscore_runs_and_respects_filter():
    h, V = make()
    hits = h.query(V[2], "auth module", k=3, fusion="zscore", where={"g": 0})
    assert hits and all(x.metadata["g"] == 0 for x in hits)


def test_zscore_alpha_extremes():
    h, V = make()
    dense_only = h.query(V[1], "auth", k=1, fusion="zscore", alpha=1.0)[0].id
    assert dense_only == "d1"
    lex_only = h.query(V[1], "auth", k=2, fusion="zscore", alpha=0.0)
    assert lex_only[0].id in {"d2", "d4"}


def test_zscore_outlier_robust():
    h, V = make()
    a = [x.id for x in h.query(V[0], "fox", k=3, fusion="zscore")]
    assert a[0] == "d0"


def test_bad_fusion():
    h, V = make()
    with pytest.raises(ValueError):
        h.query(V[0], "x", fusion="nope")
