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
