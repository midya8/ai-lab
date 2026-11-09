import numpy as np
import pytest

from vecdb.pq import ProductQuantizer


def data(n=600, d=16, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_adc_equals_distance_to_decoded_vector():
    X = data()
    pq = ProductQuantizer(16, 4, nbits=5).fit(X)
    codes = pq.encode(X[:50])
    q = X[100]
    table = pq.adc_table(q)
    adc = pq.adc_distances(table, codes)
    direct = ((pq.decode(codes) - q) ** 2).sum(1)
    assert np.allclose(adc, direct, atol=1e-3)


def test_more_bits_or_subquantizers_reduce_error():
    X = data()
    e3 = ProductQuantizer(16, 4, nbits=3).fit(X).mse(X)
    e6 = ProductQuantizer(16, 4, nbits=6).fit(X).mse(X)
    e6m8 = ProductQuantizer(16, 8, nbits=6).fit(X).mse(X)
    assert e6 < e3 and e6m8 < e6
    assert e3 < float(np.mean(np.sum(X ** 2, axis=1)))  # beats quantising everything to zero


def test_codes_are_m_bytes_and_decode_shape():
    X = data(300)
    pq = ProductQuantizer(16, 4, nbits=8).fit(X)
    c = pq.encode(X)
    assert c.dtype == np.uint8 and c.shape == (300, 4) and pq.decode(c).shape == (300, 16)
    assert pq.code_bytes() == 4


def test_validation():
    with pytest.raises(ValueError):
        ProductQuantizer(10, 3)
    with pytest.raises(ValueError):
        ProductQuantizer(16, 4, nbits=9)
    pq = ProductQuantizer(16, 4, nbits=8)
    with pytest.raises(ValueError):
        pq.fit(data(100))            # fewer points than centroids
    with pytest.raises(RuntimeError):
        pq.encode(data(5))
