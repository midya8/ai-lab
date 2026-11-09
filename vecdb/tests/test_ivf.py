import numpy as np
import pytest

from vecdb import FlatIndex, IVFIndex, clustered, queries_like, recall_at_k


@pytest.fixture(scope="module")
def world():
    X, _ = clustered(1500, 16, n_clusters=20, spread=0.6, seed=3)
    Q = queries_like(X, 40, seed=4)
    flat = FlatIndex(16)
    flat.add(X)
    gt, gd = flat.search(Q, 10)
    return X, Q, gt, gd


def test_full_probe_ivf_flat_is_exact(world):
    X, Q, gt, gd = world
    ivf = IVFIndex(16, nlist=24)
    ivf.add(X)
    ivf.train()
    ids, d = ivf.search(Q, 10, nprobe=24)
    assert np.array_equal(ids, gt) and np.allclose(d, gd, atol=1e-3)


def test_recall_grows_with_nprobe_and_scans_less(world):
    X, Q, gt, _ = world
    ivf = IVFIndex(16, nlist=24)
    ivf.add(X)
    ivf.train()
    rec, scanned = [], []
    for npb in (1, 3, 8, 24):
        ids, _ = ivf.search(Q, 10, nprobe=npb)
        rec.append(recall_at_k(ids, gt))
        scanned.append(ivf.last_stats["scanned"])
    assert rec == sorted(rec) and rec[-1] == 1.0 and rec[0] < 1.0
    assert scanned == sorted(scanned) and scanned[0] < 0.3 * len(X)


def test_add_after_train_and_buffering(world):
    X, Q, gt, _ = world
    ivf = IVFIndex(16, nlist=24)
    ivf.add(X[:1000])                      # buffered
    with pytest.raises(RuntimeError):
        ivf.search(Q, 5)
    ivf.train()
    assert ivf.list_sizes().sum() == 1000
    rows = ivf.add(X[1000:])               # added to trained lists
    assert rows[0] == 1000 and ivf.list_sizes().sum() == 1500
    ids, _ = ivf.search(Q, 10, nprobe=24)
    assert np.array_equal(ids, gt)


def test_delete_and_mask(world):
    X, Q, gt, _ = world
    ivf = IVFIndex(16, nlist=24)
    ivf.add(X)
    ivf.train()
    ivf.remove(gt[0, :3])
    ids, _ = ivf.search(Q[:1], 10, nprobe=24)
    assert not set(ids[0]) & set(gt[0, :3])
    mask = np.arange(len(X)) % 3 == 0
    ids, _ = ivf.search(Q, 10, nprobe=24, mask=mask)
    assert all(r % 3 == 0 for r in ids.ravel() if r >= 0)


def test_pq_compresses_and_refine_recovers_recall(world):
    X, Q, gt, _ = world
    plain = IVFIndex(16, nlist=24)
    plain.add(X); plain.train()
    pqi = IVFIndex(16, nlist=24, pq_m=4, pq_bits=6, keep_raw=True)
    pqi.add(X); pqi.train()
    ids_pq, _ = pqi.search(Q, 10, nprobe=24)
    ids_rf, d_rf = pqi.search(Q, 10, nprobe=24, refine=8)
    r_pq, r_rf = recall_at_k(ids_pq, gt), recall_at_k(ids_rf, gt)
    assert r_pq < 1.0 and r_rf > r_pq and r_rf >= 0.95
    code_bytes = sum(a.nbytes for a in pqi._data)
    assert code_bytes == 1500 * 4 and code_bytes < sum(a.nbytes for a in plain._data) / 10
    # refined distances are exact distances to raw vectors
    exact = ((Q[0] - X[ids_rf[0]]) ** 2).sum(1)
    assert np.allclose(d_rf[0], exact, atol=1e-3)


def test_refine_requires_raw_and_ip_rejects_pq(world):
    X, Q, _, _ = world
    pqi = IVFIndex(16, nlist=8, pq_m=4, pq_bits=5)
    pqi.add(X); pqi.train()
    with pytest.raises(ValueError):
        pqi.search(Q, 5, refine=4)
    with pytest.raises(ValueError):
        IVFIndex(16, metric="ip", pq_m=4)


def test_training_needs_enough_points():
    ivf = IVFIndex(4, nlist=10)
    ivf.add(np.random.default_rng(0).normal(size=(5, 4)))
    with pytest.raises(ValueError):
        ivf.train()
    with pytest.raises(ValueError):
        IVFIndex(4, nlist=10).train()


def test_reconstruct(world):
    X, _, _, _ = world
    ivf = IVFIndex(16, nlist=8)
    ivf.add(X[:400]); ivf.train()
    assert np.allclose(ivf.reconstruct([5, 77]), X[[5, 77]])
    pqi = IVFIndex(16, nlist=8, pq_m=8, pq_bits=6)
    pqi.add(X[:400]); pqi.train()
    err = np.linalg.norm(pqi.reconstruct([5]) - X[5]); assert 0 < err < np.linalg.norm(X[5])
