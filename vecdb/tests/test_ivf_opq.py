import numpy as np
import pytest

from vecdb import FlatIndex, IVFIndex, load_index, save_index, recall_at_k


def rotated_data(n=3000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.linspace(3.0, 0.2, d)
    return ((rng.normal(size=(n, d)) * scales) @ Q.T).astype(np.float32)


def build(opq, X):
    return IVFIndex(32, nlist=8, pq_m=8, pq_bits=5, opq=opq, opq_rounds=4).train(X)


def test_opq_requires_pq():
    with pytest.raises(ValueError):
        IVFIndex(8, nlist=4, opq=True)


def test_opq_lowers_residual_reconstruction_error():
    X = rotated_data()
    errs = {}
    for opq in (False, True):
        idx = build(opq, X)
        idx.add(X)
        errs[opq] = float(np.mean(np.sum((idx.reconstruct(np.arange(len(X))) - X) ** 2, axis=1)))
    assert errs[True] < errs[False]


def test_opq_ivf_search_and_recall_not_worse():
    X = rotated_data()
    Q = X[:60] + 0.05 * np.random.default_rng(1).normal(size=(60, 32)).astype(np.float32)
    flat = FlatIndex(32)
    flat.add(X)
    truth, _ = flat.search(Q, 10)
    rec = {}
    for opq in (False, True):
        idx = build(opq, X)
        idx.add(X)
        got, _ = idx.search(Q, 10, nprobe=8)
        rec[opq] = recall_at_k(got, truth)
    assert rec[True] >= rec[False] - 0.02


def test_opq_ivf_persistence_roundtrip(tmp_path):
    X = rotated_data(1200)
    idx = build(True, X)
    idx.add(X)
    save_index(idx, str(tmp_path / "i"))
    back = load_index(str(tmp_path / "i"))
    assert back.opq and np.allclose(back.pq.R, idx.pq.R)
    a, da = idx.search(X[:5], 5, nprobe=4)
    b, db = back.search(X[:5], 5, nprobe=4)
    assert (a == b).all() and np.allclose(da, db, atol=1e-4)
