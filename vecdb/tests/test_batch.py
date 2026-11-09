import numpy as np
import pytest

from vecdb import FlatIndex, HNSWIndex, IVFIndex, SQ8Index
from vecdb.batch import parallel_search


def setup(n=1500, d=24, nq=300):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(n, d)).astype(np.float32)
    Q = rng.normal(size=(nq, d)).astype(np.float32)
    return X, Q


def test_flat_matches_serial_and_order():
    X, Q = setup()
    idx = FlatIndex(24)
    idx.add(X)
    a = idx.search(Q, 5)
    b = parallel_search(idx, Q, 5, workers=4, shard=40)
    assert np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])


def test_sq8_and_mask_kwarg():
    X, Q = setup()
    idx = SQ8Index(24)
    idx.add(X)
    mask = np.arange(len(X)) % 2 == 0
    a = idx.search(Q, 5, mask=mask)
    b = parallel_search(idx, Q, 5, workers=3, shard=50, mask=mask)
    assert np.array_equal(a[0], b[0])
    assert (b[0] % 2 == 0).all()


def test_ivf_and_hnsw_identical():
    X, Q = setup(n=800, nq=120)
    for idx in (IVFIndex(24, nlist=8), HNSWIndex(24)):
        if hasattr(idx, "train"):
            idx.train(X)
        idx.add(X)
        a = idx.search(Q, 5)
        b = parallel_search(idx, Q, 5, workers=4, shard=25)
        assert np.array_equal(a[0], b[0])


def test_small_batch_and_validation():
    X, Q = setup()
    idx = FlatIndex(24)
    idx.add(X)
    assert parallel_search(idx, Q[:3], 4, workers=8)[0].shape == (3, 4)
    with pytest.raises(ValueError):
        parallel_search(idx, Q, 4, workers=0)
