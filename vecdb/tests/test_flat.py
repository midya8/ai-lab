import numpy as np
import pytest

from vecdb import FlatIndex


def brute(X, Q, k, metric):
    if metric == "l2":
        D = ((Q[:, None] - X[None]) ** 2).sum(-1)
    elif metric == "ip":
        D = -(Q @ X.T)
    else:
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        D = 1 - Qn @ Xn.T
    return np.argsort(D, axis=1, kind="stable")[:, :k], np.sort(D, axis=1)[:, :k]


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_matches_numpy_bruteforce(metric):
    rng = np.random.default_rng(0)
    X, Q = rng.normal(size=(300, 12)).astype(np.float32), rng.normal(size=(20, 12)).astype(np.float32)
    idx = FlatIndex(12, metric)
    rows = idx.add(X)
    assert rows.tolist() == list(range(300))
    ids, d = idx.search(Q, 7)
    want_i, want_d = brute(X, Q, 7, metric)
    assert np.array_equal(ids, want_i)
    assert np.allclose(d, want_d, atol=1e-3)


def test_k_larger_than_n_pads_and_empty_index():
    idx = FlatIndex(3)
    ids, d = idx.search([[0, 0, 0]], 4)
    assert (ids == -1).all() and np.isinf(d).all()
    idx.add([[1, 0, 0], [0, 1, 0]])
    ids, d = idx.search([[1, 0, 0]], 4)
    assert ids[0].tolist() == [0, 1, -1, -1]


def test_delete_and_mask_filters():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4)).astype(np.float32)
    idx = FlatIndex(4)
    idx.add(X)
    first, _ = idx.search(X[:1], 1)
    assert first[0, 0] == 0
    idx.remove([0])
    nxt, _ = idx.search(X[:1], 1)
    assert nxt[0, 0] != 0 and idx.n_live == 49
    mask = np.arange(50) % 2 == 0
    ids, _ = idx.search(X[:5], 10, mask=mask)
    assert all(r % 2 == 0 for r in ids.ravel() if r >= 0)
    assert 0 not in ids
    with pytest.raises(ValueError):
        idx.search(X[:1], 3, mask=np.ones(10, dtype=bool))
    with pytest.raises(IndexError):
        idx.remove([99])


def test_dimension_and_k_validation():
    idx = FlatIndex(4)
    with pytest.raises(ValueError):
        idx.add(np.zeros((2, 5)))
    idx.add(np.eye(4))
    with pytest.raises(ValueError):
        idx.search(np.zeros((1, 4)), 0)
