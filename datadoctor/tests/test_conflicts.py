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


def test_near_conflicts_finds_close_disagreeing_pairs():
    from datadoctor.conflicts import near_conflicts
    X = [[0, 0], [0.1, 0], [5, 5], [5.05, 5], [9, 9]]
    y = [0, 1, 0, 0, 1]
    r = near_conflicts(X, y, radius=0.2)
    assert r["n_pairs"] == 1 and r["pairs"][0][:2] == (0, 1)
    assert abs(r["pairs"][0][2] - 0.1) < 1e-6 and r["rows"] == [0, 1]


def test_near_conflicts_matches_bruteforce_across_blocks():
    import numpy as np
    from datadoctor.conflicts import near_conflicts
    rng = np.random.default_rng(0)
    X = rng.normal(size=(700, 3)); y = rng.integers(0, 3, 700)
    r = near_conflicts(X, y, radius=0.4)
    exp = {(i, j) for i in range(700) for j in range(i + 1, 700)
           if y[i] != y[j] and np.linalg.norm(X[i] - X[j]) <= 0.4}
    assert {p[:2] for p in r["pairs"]} == exp


def test_near_conflicts_validation():
    import pytest
    from datadoctor.conflicts import near_conflicts
    with pytest.raises(ValueError):
        near_conflicts([[0]], [0], radius=-1)
    with pytest.raises(ValueError):
        near_conflicts([[0], [1]], [0, 1], radius=1, max_rows=1)
