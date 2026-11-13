import numpy as np
import pytest
from datadoctor.conflicts import label_conflicts


def test_finds_minority():
    X = [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4]]
    y = [0, 0, 1, 5, 5]
    r = label_conflicts(X, y)
    assert r["n_groups"] == 1 and r["n_rows"] == 3
    g = r["groups"][0]
    assert g["majority"] == 0 and g["minority_rows"] == [2]


def test_tie_and_clean():
    r = label_conflicts([[0], [0]], [1, 2])
    assert r["groups"][0]["majority"] is None and r["groups"][0]["minority_rows"] == []
    assert label_conflicts([[0], [1]], [1, 2])["n_groups"] == 0


def test_rounding_and_validation():
    assert label_conflicts([[0.0000001], [0.0]], [0, 1])["n_groups"] == 1
    with pytest.raises(ValueError):
        label_conflicts([1, 2], [0, 1])
