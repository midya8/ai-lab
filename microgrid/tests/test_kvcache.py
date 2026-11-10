import numpy as np
import pytest
from microgrid import MiniGPT
from microgrid.lm import generate
from microgrid.kvcache import prefill, step, forward_new, KVCache, generate_cached


def model(seed=0, ctx=16):
    m = MiniGPT(11, d=16, heads=4, layers=2, ctx=ctx, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for p in m.parameters():  # make LN/bias non-trivial
        p.data += rng.normal(0, 0.05, p.data.shape)
    return m


def test_prefill_matches_autodiff_forward():
    m = model()
    ids = np.array([1, 5, 2, 7, 3, 3])
    lg, _ = prefill(m, ids)
    assert np.allclose(lg, m(ids[None]).data[0, -1], atol=1e-9)


def test_incremental_equals_full_at_every_position():
    m = model(2)
    ids = np.random.default_rng(0).integers(0, 11, 12)
    full = m(ids[None]).data[0]
    cache = KVCache(m)
    for t, tok in enumerate(ids):
        assert np.allclose(forward_new(m, [tok], cache)[0], full[t], atol=1e-9)


def test_chunked_prefill_equals_single():
    m = model(3)
    ids = np.arange(10) % 11
    a = forward_new(m, ids, KVCache(m))
    c = KVCache(m)
    b = np.concatenate([forward_new(m, ids[:4], c), forward_new(m, ids[4:], c)])
    assert np.allclose(a, b, atol=1e-9)


def test_generate_matches_reference_incl_sliding_window():
    m = model(4, ctx=8)
    chars = list("abcdefghijk")
    for n in (5, 30):  # 30 > ctx forces the rebuild path
        assert generate_cached(m, chars, [1, 2], n, 8, seed=7) == generate(m, chars, [1, 2], n, 8, seed=7)


def test_overflow_asserts():
    m = model(ctx=4)
    with pytest.raises(AssertionError):
        forward_new(m, np.arange(5), KVCache(m))


from microgrid.kvcache import QuantKVCache, _sample, _q8


def test_q8_roundtrip_error_bounded():
    x = np.random.default_rng(0).normal(size=(4, 9, 16))
    q, s = _q8(x)
    assert q.dtype == np.int8
    assert np.abs(q * s - x).max() <= s.max() / 2 + 1e-6


def test_q8_zero_rows_safe():
    q, s = _q8(np.zeros((2, 3, 4)))
    assert np.all(q == 0) and np.all(np.isfinite(s))


def test_quant_cache_close_to_exact_and_smaller():
    m = model(4)
    ids = np.random.default_rng(1).integers(0, 11, 12)
    lg_e, ce = prefill(m, ids)
    lg_q, cq = prefill(m, ids, QuantKVCache)
    assert cq.length == ce.length == 12
    assert np.abs(lg_e - lg_q).max() < 0.05
    assert cq.nbytes() < ce.nbytes() / 3


def test_quant_cache_incremental_matches_chunked():
    m = model(5)
    ids = np.arange(9) % 11
    a = forward_new(m, ids, QuantKVCache(m))
    c = QuantKVCache(m)
    b = np.concatenate([forward_new(m, ids[:5], c), forward_new(m, ids[5:], c)])
    assert np.abs(a - b).max() < 0.05  # per-token scales: only attention-input noise differs


def test_top_k_one_is_greedy():
    rng = np.random.default_rng(0)
    lg = np.array([0.1, 3.0, 2.9, -1.0])
    assert all(_sample(lg, 1.0, rng, top_k=1) == 1 for _ in range(20))


def test_top_k_restricts_support():
    rng = np.random.default_rng(0)
    lg = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    seen = {_sample(lg, 5.0, rng, top_k=2) for _ in range(300)}
    assert seen <= {3, 4} and len(seen) == 2


def test_top_p_restricts_support():
    rng = np.random.default_rng(0)
    lg = np.log(np.array([0.6, 0.3, 0.07, 0.03]))
    seen = {_sample(lg, 1.0, rng, top_p=0.85) for _ in range(300)}
    assert seen == {0, 1}


def test_default_sampling_unchanged_and_filters_deterministic():
    m = model(6)
    chars = list("abcdefghijk")
    a = generate_cached(m, chars, [1, 2], 20, 16, seed=3)
    b = generate_cached(m, chars, [1, 2], 20, 16, seed=3, top_k=None, top_p=None)
    assert a == b
    c = generate_cached(m, chars, [1, 2], 20, 16, seed=3, top_k=3, cache_cls=QuantKVCache)
    d = generate_cached(m, chars, [1, 2], 20, 16, seed=3, top_k=3, cache_cls=QuantKVCache)
    assert c == d and len(c) == 22


def test_cached_nll_matches_autodiff_loss_and_quant_close():
    import numpy as np
    from microgrid import MiniGPT
    from microgrid.kvcache import cached_nll, QuantKVCache
    from microgrid.lm import lm_loss
    m = MiniGPT(11, d=16, heads=2, layers=2, ctx=12, seed=3)
    ids = np.random.default_rng(0).integers(0, 11, 13)  # one window of 12 predictions
    exact, n = cached_nll(m, ids)
    assert n == 12
    ref = float(lm_loss(m, ids[None, :12], ids[None, 1:13]).data)
    assert abs(exact - ref) < 1e-9
    q, _ = cached_nll(m, ids, QuantKVCache)
    assert abs(q - exact) < 0.05
