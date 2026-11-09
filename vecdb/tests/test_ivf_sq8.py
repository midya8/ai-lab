import numpy as np
import pytest

from vecdb import FlatIndex, IVFIndex, load_index, save_index, recall_at_k


def data(n=3000, d=32, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_sq8_excludes_pq():
    with pytest.raises(ValueError):
        IVFIndex(8, nlist=4, pq_m=2, sq8=True)


def test_reconstruction_error_within_half_step():
    X = data(2000)  # below max_train, so the fitted range covers every vector
    idx = IVFIndex(32, nlist=8, sq8=True).train(X)
    idx.add(X)
    err = np.abs(idx.reconstruct(np.arange(len(X))) - X).max()
    assert err <= float(idx.sq_step.max()) / 2 + 1e-5


def test_memory_is_about_quarter_of_flat():
    X = data()
    flat = IVFIndex(32, nlist=8).train(X)
    sq = IVFIndex(32, nlist=8, sq8=True).train(X)
    flat.add(X); sq.add(X)
    payload_flat = sum(a.nbytes for a in flat._data)
    payload_sq = sum(a.nbytes for a in sq._data)
    assert payload_sq * 4 == payload_flat


def test_recall_close_to_ivf_flat_and_refine_helps():
    X = data()
    Q = X[:60] + 0.05 * np.random.default_rng(1).normal(size=(60, 32)).astype(np.float32)
    f = FlatIndex(32); f.add(X)
    truth, _ = f.search(Q, 10)
    flat = IVFIndex(32, nlist=8).train(X); flat.add(X)
    sq = IVFIndex(32, nlist=8, sq8=True, keep_raw=True).train(X); sq.add(X)
    r_flat = recall_at_k(flat.search(Q, 10, nprobe=8)[0], truth)
    r_sq = recall_at_k(sq.search(Q, 10, nprobe=8)[0], truth)
    r_ref = recall_at_k(sq.search(Q, 10, nprobe=8, refine=4)[0], truth)
    assert r_flat > 0.99
    assert r_sq >= 0.9
    assert r_ref >= r_sq and r_ref > 0.99


def test_refine_needs_keep_raw():
    X = data(500)
    idx = IVFIndex(32, nlist=4, sq8=True).train(X); idx.add(X)
    with pytest.raises(ValueError):
        idx.search(X[:2], 5, refine=2)


def test_delete_and_roundtrip(tmp_path):
    X = data(1000)
    idx = IVFIndex(32, nlist=4, sq8=True, keep_raw=True).train(X); idx.add(X)
    idx.remove([0, 1])
    p = str(tmp_path / "i.npz")
    save_index(idx, p)
    back = load_index(p)
    assert back.sq8
    a, da = idx.search(X[:5], 5, nprobe=4)
    b, db = back.search(X[:5], 5, nprobe=4)
    assert (a == b).all() and np.allclose(da, db, atol=1e-5)
    assert 0 not in b and 1 not in b
