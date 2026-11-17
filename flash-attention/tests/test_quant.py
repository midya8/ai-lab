import numpy as np
import pytest

from flash_attention import OutOfBlocks, PagedKVCache, QuantPagedKVCache, naive_attention, quant


def rnd(*shape, seed=0, scale=1.0):
    return np.random.default_rng(seed).normal(size=shape) * scale


@pytest.mark.parametrize("bits", [2, 4, 8])
@pytest.mark.parametrize("group", [None, 16, 4])
def test_error_bounded_by_half_scale(bits, group):
    x = rnd(30, 3, 16, seed=bits, scale=5.0)
    codes, sc = quant.quantize(x, bits, group)
    assert codes.min() >= -quant.qmax(bits) and codes.max() <= quant.qmax(bits)
    xh = quant.dequantize(codes, sc, group)
    g = 16 if group is None else group
    bound = np.repeat(sc.astype(np.float64), g, axis=-1) / 2
    assert np.all(np.abs(x - xh) <= bound + 1e-12)


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_pack_unpack_roundtrip_exhaustive(bits):
    qm = quant.qmax(bits)
    vals = np.arange(-qm, qm + 1, dtype=np.int8)
    d = 8 * (8 // bits)
    codes = np.resize(vals, (5, 3, d)).astype(np.int8)
    packed = quant.pack(codes, bits)
    assert packed.shape == (5, 3, d * bits // 8) and packed.dtype == np.uint8
    assert np.array_equal(quant.unpack(packed, bits, d), codes)


def test_zero_vector_and_constant_vector():
    codes, sc = quant.quantize(np.zeros((2, 8)), 4)
    assert np.all(codes == 0) and np.all(quant.dequantize(codes, sc) == 0)
    codes, sc = quant.quantize(np.full((1, 8), 3.0), 4)
    assert np.allclose(quant.dequantize(codes, sc), 3.0, atol=float(sc.max()) / 2)


def test_more_bits_never_worse_and_smaller_groups_help():
    x = rnd(200, 32, seed=1)
    x[:, 0] *= 30                                 # one outlier channel
    err = lambda b, g: np.abs(x - quant.dequantize(*quant.quantize(x, b, g), g)).mean()
    assert err(8, None) < err(4, None) < err(2, None)
    assert err(4, 8) < err(4, None)               # the outlier no longer sets everyone's scale


def test_quantisation_is_per_token_so_appends_do_not_disturb_old_tokens():
    x = rnd(10, 16, seed=2)
    a = quant.quantize(x[:4], 4, 8)
    b = quant.quantize(x, 4, 8)
    assert np.array_equal(a[0], b[0][:4]) and np.array_equal(a[1], b[1][:4])


def test_scale_overflow_is_reported():
    with pytest.raises(OverflowError):
        quant.quantize(np.array([[1e9, 0, 0, 0]]), 4)
    codes, sc = quant.quantize(np.array([[1e9, 0, 0, 0]]), 4, scale_dtype=np.float32)
    assert np.isfinite(quant.dequantize(codes, sc)).all()


def test_bad_arguments():
    with pytest.raises(ValueError):
        quant.qmax(3)
    with pytest.raises(ValueError):
        quant.quantize(np.zeros((2, 10)), 4, 4)           # 4 does not divide 10
    with pytest.raises(ValueError):
        quant.pack(np.zeros((2, 3), dtype=np.int8), 4)    # odd width cannot pack 2/byte


def test_bytes_per_vector_matches_actual_arrays():
    codes, sc = quant.quantize(rnd(1, 64), 4, 16)
    assert quant.bytes_per_vector(64, 4, 16) == quant.pack(codes, 4).nbytes + sc.nbytes == 32 + 8


# ---- quantised paged cache -------------------------------------------------------------------
def toks(T, Hkv=2, d=8, seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(T, Hkv, d)), r.normal(size=(T, Hkv, d))


def test_gather_matches_quantise_dequantise_of_the_tokens():
    c = QuantPagedKVCache(16, 4, 2, 8, k_bits=4, v_bits=8, group_size=4)
    s = c.new_sequence()
    k, v = toks(11, seed=3)
    c.append(s, k[:5], v[:5])
    c.append(s, k[5:], v[5:])
    K, V = c.gather(s)
    kk = quant.dequantize(*quant.quantize(k, 4, 4), 4)
    vv = quant.dequantize(*quant.quantize(v, 8, 4), 4)
    assert np.array_equal(K.transpose(1, 0, 2), kk) and np.array_equal(V.transpose(1, 0, 2), vv)


def test_8bit_attention_close_to_exact_and_error_shrinks_with_bits():
    k, v = toks(40, seed=4)
    q = np.random.default_rng(5).normal(size=(2, 3, 8))
    ref = naive_attention(q, k.transpose(1, 0, 2), v.transpose(1, 0, 2), causal=True)[0]
    errs = []
    for bits in (2, 4, 8):
        c = QuantPagedKVCache(32, 4, 2, 8, k_bits=bits, v_bits=bits)
        s = c.new_sequence()
        c.append(s, k, v)
        errs.append(np.abs(c.attend(s, q)[0] - ref).max())
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.03


def test_quant_attention_equals_exact_attention_over_dequantised_cache():
    """Attention over the quantised blocks is *exact* given the dequantised values: the only
    error is the quantisation of K and V, not the blockwise online softmax."""
    c = QuantPagedKVCache(32, 4, 2, 8, k_bits=4, v_bits=4, group_size=4)
    s = c.new_sequence()
    k, v = toks(37, seed=6)
    c.append(s, k, v)
    K, V = c.gather(s)
    q = np.random.default_rng(7).normal(size=(4, 1, 8))
    o, l = c.attend(s, q, window=9)
    o2, l2, _ = naive_attention(q, K, V, causal=True, window=9)
    assert np.allclose(o, o2, atol=1e-12) and np.allclose(l, l2, atol=1e-12)


def test_fork_and_copy_on_write_with_quantised_blocks():
    c = QuantPagedKVCache(16, 4, 2, 8, k_bits=4, v_bits=4)
    a = c.new_sequence()
    k, v = toks(6, seed=8)
    c.append(a, k, v)
    b = c.fork(a)
    before_a = [x.copy() for x in c.gather(a)]
    k2, v2 = toks(1, seed=9)
    c.append(b, k2, v2)                       # tail block shared -> copy-on-write
    assert c.copies == 1
    for x, y in zip(c.gather(a), before_a):
        assert np.array_equal(x, y)           # parent untouched
    Kb, _ = c.gather(b)
    assert np.array_equal(Kb[:, :6], c.gather(a)[0])
    assert Kb.shape[1] == 7


def test_out_of_blocks_does_not_half_apply():
    c = QuantPagedKVCache(2, 4, 2, 8, k_bits=2, v_bits=2)
    s = c.new_sequence()
    c.append(s, *toks(8))
    with pytest.raises(OutOfBlocks):
        c.append(s, *toks(1))
    assert c.lengths[s] == 8


def test_memory_accounting_is_real_and_compression_ratios_hold():
    fp = PagedKVCache(10, 16, 4, 64, dtype=np.float16)
    assert fp.bytes_per_token() == 2 * 4 * 64 * 2          # K+V, 4 heads, d=64, 2 bytes
    for bits, g, ratio in ((8, 64, 2 * 64 / (64 + 2)), (4, 32, 2 * 64 / (32 + 4))):
        c = QuantPagedKVCache(10, 16, 4, 64, k_bits=bits, v_bits=bits, group_size=g)
        per_vec = quant.bytes_per_vector(64, bits, g)
        assert c.bytes_per_token() == 2 * 4 * per_vec
        assert fp.bytes_per_token() / c.bytes_per_token() == pytest.approx(ratio)
    # 2-bit with small groups pays visibly for scales
    c2 = QuantPagedKVCache(10, 16, 4, 64, k_bits=2, v_bits=2, group_size=16)
    assert fp.bytes_per_token() / c2.bytes_per_token() < 8


def test_head_dim_must_fit_packing():
    with pytest.raises(ValueError):
        QuantPagedKVCache(4, 4, 1, 6, k_bits=2, v_bits=2)     # 6 not a multiple of 4


def test_unquantised_cache_unchanged_by_refactor():
    c = PagedKVCache(8, 4, 2, 8, dtype=np.float32)
    s = c.new_sequence()
    k, v = toks(9, seed=1)
    c.append(s, k, v)
    K, _ = c.gather(s)
    assert K.dtype == np.float32 and np.allclose(K.transpose(1, 0, 2), k, atol=1e-6)
