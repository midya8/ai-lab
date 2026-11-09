import numpy as np
import pytest

from vecdb import FlatIndex, SQ8Index, recall_at_k


def data(n=2000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    return X, X[:min(50, n)] + 0.1 * rng.normal(size=(min(50, n), d)).astype(np.float32)


def test_roundtrip_error_bounded():
    X, _ = data()
    idx = SQ8Index(32)
    idx.add(X)
    err = np.abs(idx.reconstruct(np.arange(len(X))) - X).max()
    assert err <= idx.max_abs_error() + 1e-5


def test_constant_dimension_exact():
    X = np.random.default_rng(1).normal(size=(50, 4)).astype(np.float32)
    X[:, 2] = 3.5
    idx = SQ8Index(4)
    idx.add(X)
    assert np.allclose(idx.reconstruct(np.arange(50))[:, 2], 3.5)


def test_memory_is_quarter():
    X, _ = data()
    f, s = FlatIndex(32), SQ8Index(32)
    f.add(X); s.add(X)
    assert s.memory_bytes() * 4 == f.memory_bytes()


@pytest.mark.parametrize("metric", ["l2", "cosine", "ip"])
def test_recall_high_and_refine_helps(metric):
    X, Q = data()
    f, s = FlatIndex(32, metric), SQ8Index(32, metric, keep_raw=True)
    f.add(X); s.add(X)
    truth, _ = f.search(Q)
    r0 = recall_at_k(s.search(Q, 10)[0], truth)
    r1 = recall_at_k(s.search(Q, 10, refine=4)[0], truth)
    assert r0 >= 0.9
    assert r1 >= r0 and r1 >= 0.99


def test_refine_requires_raw():
    X, Q = data(100)
    s = SQ8Index(32)
    s.add(X)
    with pytest.raises(ValueError):
        s.search(Q, 5, refine=2)


def test_delete_mask_and_padding():
    X, Q = data(20)
    s = SQ8Index(32)
    s.add(X)
    s.remove([0, 1])
    ids, _ = s.search(Q[:3], 25)
    assert not np.isin(ids, [0, 1]).any()
    assert (ids[:, 18:] == -1).all()
    mask = np.zeros(20, dtype=bool); mask[5:8] = True
    ids, _ = s.search(Q[:2], 3, mask=mask)
    assert set(ids.ravel().tolist()) <= {5, 6, 7}


def test_out_of_range_values_clipped_not_crashing():
    X, _ = data(100)
    s = SQ8Index(32)
    s.add(X)
    s.add(X[:3] * 10)
    assert s.ntotal == 103


@pytest.mark.parametrize("keep_raw", [False, True])
def test_save_load_roundtrip(tmp_path, keep_raw):
    from vecdb import load_index, save_index
    X, Q = data(500, 16)
    idx = SQ8Index(16, keep_raw=keep_raw)
    idx.add(X)
    idx.remove([3, 7])
    p = str(tmp_path / "i.npz")
    save_index(idx, p)
    j = load_index(p)
    assert isinstance(j, SQ8Index) and j.keep_raw == keep_raw
    kw = {"refine": 2} if keep_raw else {}
    a, da = idx.search(Q, 5, **kw)
    b, db = j.search(Q, 5, **kw)
    assert (a == b).all() and np.allclose(da, db)
    assert j.deleted[3] and j.deleted[7]


def test_save_load_untrained(tmp_path):
    from vecdb import load_index, save_index
    p = str(tmp_path / "e.npz")
    save_index(SQ8Index(8), p)
    j = load_index(p)
    assert not j.is_trained and j.ntotal == 0
