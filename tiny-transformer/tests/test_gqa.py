import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, generate_cached  # noqa: E402
from tinygpt.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from tinygpt.kvcache import _prefill  # noqa: E402
from tinygpt.model import expand_kv, kv_cache_bytes, reduce_kv  # noqa: E402
from tinygpt.tokenizer import CharTokenizer  # noqa: E402


def make(n_kv_head, n_head=4, **kw):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=n_head, n_embd=16,
                    n_kv_head=n_kv_head, **kw)
    m = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    return m, rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))


def test_expand_and_reduce_are_adjoint():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 2, 5, 3))
    y = rng.standard_normal((2, 6, 5, 3))
    np.testing.assert_allclose((expand_kv(x, 3) * y).sum(), (x * reduce_kv(y, 3)).sum())
    # head h of the expanded tensor is kv head h // g
    e = expand_kv(x, 3)
    for h in range(6):
        np.testing.assert_array_equal(e[:, h], x[:, h // 3])


def test_default_is_plain_mha_and_shapes_shrink():
    assert GPTConfig(11, n_head=4, n_embd=16).kv_heads == 4
    mha, _, _ = make(0)
    gqa, _, _ = make(2)
    mqa, _, _ = make(1)
    assert mha.params["h0.qkv_W"].shape == (16, 48)  # unchanged layout
    assert gqa.params["h0.qkv_W"].shape == (16, 16 + 2 * 8)
    assert mqa.params["h0.qkv_W"].shape == (16, 16 + 2 * 4)
    assert mqa.num_params() < gqa.num_params() < mha.num_params()


def test_explicit_full_kv_heads_equals_default():
    a, idx, tgt = make(0)
    b, _, _ = make(4)
    for k in a.params:
        b.params[k][...] = a.params[k]
    np.testing.assert_allclose(a.forward(idx)[0], b.forward(idx)[0])


def test_invalid_kv_heads_rejected():
    with pytest.raises(AssertionError):
        GPT(GPTConfig(11, 8, 1, 4, 16, n_kv_head=3))
    with pytest.raises(AssertionError):
        GPT(GPTConfig(11, 8, 1, 4, 16, n_kv_head=8))


def test_mqa_equals_mha_with_tied_kv_heads():
    """An MHA model whose per-head K/V weights are copies of one head must match MQA exactly."""
    mqa, idx, tgt = make(1)
    mha, _, _ = make(0)
    C, D = 16, 4
    for k, v in mqa.params.items():
        if k.endswith("qkv_W") or k.endswith("qkv_b"):
            continue
        mha.params[k][...] = v
    for i in range(2):
        W, b = mqa.params[f"h{i}.qkv_W"], mqa.params[f"h{i}.qkv_b"]
        Wq, Wk, Wv = W[:, :C], W[:, C:C + D], W[:, C + D:]
        bq, bk, bv = b[:C], b[C:C + D], b[C + D:]
        mha.params[f"h{i}.qkv_W"][...] = np.concatenate([Wq, np.tile(Wk, 4), np.tile(Wv, 4)], 1)
        mha.params[f"h{i}.qkv_b"][...] = np.concatenate([bq, np.tile(bk, 4), np.tile(bv, 4)])
    np.testing.assert_allclose(mqa.forward(idx, tgt)[0], mha.forward(idx, tgt)[0], atol=1e-12)


@pytest.mark.parametrize("kv,extra", [(2, {}), (1, {}), (2, {"rope": True}), (1, {"rope": True, "tie_weights": True})])
def test_grads_match_finite_differences(kv, extra):
    m, idx, tgt = make(kv, **extra)
    _, G = m.loss_and_grads(idx, tgt)
    rng = np.random.default_rng(0)
    for name in ["wte", "h0.qkv_W", "h1.qkv_W", "h0.qkv_b", "h1.proj_W"]:
        assert G[name].shape == m.params[name].shape
        for _ in range(10):
            ix = tuple(rng.integers(0, s) for s in m.params[name].shape)
            old = m.params[name][ix]
            m.params[name][ix] = old + 1e-6
            lp = m.forward(idx, tgt)[1]
            m.params[name][ix] = old - 1e-6
            lm = m.forward(idx, tgt)[1]
            m.params[name][ix] = old
            assert abs((lp - lm) / 2e-6 - G[name][ix]) < 1e-6, (name, ix)


def test_grads_with_dropout_masks():
    m, idx, tgt = make(2, dropout=0.2)
    _, G = m.loss_and_grads(idx, tgt, rng=np.random.default_rng(5))

    def loss():
        return m.forward(idx, tgt, rng=np.random.default_rng(5))[1]

    rng = np.random.default_rng(1)
    for name in ["h0.qkv_W", "wte"]:
        for _ in range(6):
            ix = tuple(rng.integers(0, s) for s in m.params[name].shape)
            old = m.params[name][ix]
            m.params[name][ix] = old + 1e-6
            lp = loss()
            m.params[name][ix] = old - 1e-6
            lm = loss()
            m.params[name][ix] = old
            assert abs((lp - lm) / 2e-6 - G[name][ix]) < 1e-6


def test_causality_holds():
    m, idx, _ = make(2)
    a = m.forward(idx)[0]
    idx2 = idx.copy()
    idx2[:, 4:] = (idx2[:, 4:] + 1) % 11
    b = m.forward(idx2)[0]
    np.testing.assert_allclose(a[:, :4], b[:, :4])


@pytest.mark.parametrize("kv,rope", [(2, False), (1, False), (2, True), (1, True)])
def test_kv_cache_matches_full_forward(kv, rope):
    m, idx, _ = make(kv, rope=rope)
    ctx = idx[0]
    logits, cache = _prefill(m, ctx)
    np.testing.assert_allclose(logits, m.forward(ctx[None])[0][0, -1], atol=1e-9)
    # the cache holds only the kv heads
    assert cache[0][0][0].shape == (kv, 4)
    a = generate_cached(m, ctx[:3], 10, seed=3)
    b = m.generate(ctx[:3], 10, seed=3)
    np.testing.assert_array_equal(a, b)


def test_cache_bytes_shrink_with_group_size():
    full = GPTConfig(11, 64, 4, 8, 64)
    gqa = GPTConfig(11, 64, 4, 8, 64, n_kv_head=2)
    mqa = GPTConfig(11, 64, 4, 8, 64, n_kv_head=1)
    assert kv_cache_bytes(full, 100) == 2 * 4 * 64 * 100 * 4
    assert kv_cache_bytes(gqa, 100) * 4 == kv_cache_bytes(full, 100)
    assert kv_cache_bytes(mqa, 100) * 8 == kv_cache_bytes(full, 100)


def test_checkpoint_round_trip_and_old_configs(tmp_path):
    m, idx, _ = make(2)
    tok = CharTokenizer("abcdefghijk")
    p = str(tmp_path / "c.npz")
    save_checkpoint(p, m, tok)
    m2, _ = load_checkpoint(p)
    assert m2.cfg.n_kv_head == 2
    np.testing.assert_allclose(m.forward(idx)[0], m2.forward(idx)[0], atol=1e-5)  # reload is float32
    # a config dict written before GQA existed has no n_kv_head key and means plain MHA
    from dataclasses import asdict
    d = asdict(m.cfg)
    d.pop("n_kv_head")
    assert GPTConfig(**d).kv_heads == 4
