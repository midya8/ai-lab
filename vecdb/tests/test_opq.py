import numpy as np

from vecdb.opq import OPQ
from vecdb.pq import ProductQuantizer


def correlated(n=1500, d=16, seed=0):
    """Variance concentrated in a few random directions: bad for contiguous PQ blocks."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    scales = np.linspace(3.0, 0.2, d)
    return ((rng.normal(size=(n, d)) * scales) @ Q.T).astype(np.float32)


def test_rotation_is_orthogonal():
    o = OPQ(16, 4, nbits=5).fit(correlated(), rounds=3, iters=5)
    assert np.allclose(o.R @ o.R.T, np.eye(16), atol=1e-4)


def test_opq_beats_pq_on_correlated_data():
    X = correlated()
    pq = ProductQuantizer(16, 4, nbits=5).fit(X, iters=10).mse(X)
    o = OPQ(16, 4, nbits=5).fit(X, rounds=6, iters=10)
    assert o.mse(X) < pq * 0.95


def test_never_worse_than_pq_on_training_set():
    X = np.random.default_rng(1).normal(size=(800, 16)).astype(np.float32)
    o = OPQ(16, 4, nbits=5).fit(X, rounds=3, iters=8)
    assert min(o.history) == o.history[np.argmin(o.history)]
    assert o.mse(X) <= o.history[0] * 1.001


def test_adc_matches_distance_to_decoded_in_original_space():
    X = correlated(600)
    o = OPQ(16, 4, nbits=5).fit(X, rounds=2, iters=5)
    codes = o.encode(X[:40])
    q = X[100]
    adc = o.adc_distances(o.adc_table(q), codes)
    direct = ((o.decode(codes) - q) ** 2).sum(1)
    assert np.allclose(adc, direct, atol=1e-2)


def test_untrained_encode_is_error():
    import pytest
    with pytest.raises(RuntimeError):
        OPQ(16, 4).encode(np.zeros((1, 16), np.float32))
