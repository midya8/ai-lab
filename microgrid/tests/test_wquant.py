import numpy as np
from microgrid import MiniGPT
from microgrid.wquant import quantize_array, fake_quant, quantize_model, weight_bytes
from microgrid.kvcache import cached_nll


def model(seed=0):
    return MiniGPT(13, d=16, heads=4, layers=2, ctx=12, seed=seed)


def test_codes_within_range_and_error_bound():
    w = np.random.default_rng(0).normal(size=(20, 7))
    for bits in (8, 4, 2):
        q, s = quantize_array(w, bits, axis=0)
        assert np.abs(q).max() <= 2 ** (bits - 1) - 1
        assert np.all(np.abs(q * s - w) <= s / 2 + 1e-12)


def test_zero_matrix_is_safe():
    assert np.all(fake_quant(np.zeros((3, 3)), 8, 0) == 0)


def test_error_shrinks_with_bits():
    w = np.random.default_rng(1).normal(size=(30, 30))
    errs = [np.abs(fake_quant(w, b, 0) - w).mean() for b in (2, 4, 6, 8)]
    assert errs == sorted(errs, reverse=True)


def test_per_channel_beats_per_tensor_with_outlier_column():
    w = np.random.default_rng(2).normal(size=(40, 8))
    w[:, 0] *= 50
    pc = np.abs(fake_quant(w, 4, 0) - w)[:, 1:].mean()
    pt = np.abs(fake_quant(w, 4, None) - w)[:, 1:].mean()
    assert pc < pt / 5


def test_quantize_model_copies_and_changes_weights():
    m = model()
    before = m.head.w.data.copy()
    q = quantize_model(m, 4)
    assert np.array_equal(m.head.w.data, before)
    assert not np.array_equal(q.head.w.data, before)
    assert np.array_equal(q.blocks[0].ln1.g.data, m.blocks[0].ln1.g.data)
    # each output column has at most 2^4-1 distinct values
    assert len(np.unique(q.head.w.data[:, 0])) <= 15


def test_8bit_logits_close_and_works_with_cache():
    m = model(3)
    ids = np.random.default_rng(0).integers(0, 13, 40)
    q = quantize_model(m, 8)
    assert np.abs(q(ids[None, :12]).data - m(ids[None, :12]).data).max() < 0.2
    a, _ = cached_nll(m, ids)
    b, _ = cached_nll(q, ids)
    assert abs(a - b) < 0.02


def test_bytes_ratio():
    m = model()
    full = weight_bytes(m)
    assert weight_bytes(m, 8) < full / 5
    assert weight_bytes(m, 4) < weight_bytes(m, 8)
    assert weight_bytes(m, 4, embeddings=False) > weight_bytes(m, 4)
