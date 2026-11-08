import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, generate_cached  # noqa: E402
from tinygpt.checkpoint import save_checkpoint, load_checkpoint  # noqa: E402
from tinygpt.model import rope_tables, rope_fwd, rope_bwd  # noqa: E402
from tinygpt.tokenizer import CharTokenizer  # noqa: E402


def make(**kw):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8, rope=True, **kw)
    m = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    return m, rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))


def test_rotation_preserves_norm_and_bwd_is_inverse():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 5, 8))
    cos, sin = rope_tables(5, 8)
    y = rope_fwd(x, cos, sin)
    np.testing.assert_allclose(np.linalg.norm(y, axis=-1), np.linalg.norm(x, axis=-1))
    np.testing.assert_allclose(rope_bwd(y, cos, sin), x, atol=1e-12)


def test_scores_depend_only_on_relative_position():
    rng = np.random.default_rng(1)
    q, k = rng.standard_normal(8), rng.standard_normal(8)

    def score(m, n):
        cm, sm = rope_tables(1, 8, offset=m)
        cn, sn = rope_tables(1, 8, offset=n)
        return float(rope_fwd(q[None], cm, sm)[0] @ rope_fwd(k[None], cn, sn)[0])

    assert score(5, 2) == pytest.approx(score(105, 102))
    assert score(5, 2) == pytest.approx(score(3, 0))
    assert score(5, 2) != pytest.approx(score(5, 3))


def test_position_zero_is_identity():
    x = np.random.default_rng(0).standard_normal((1, 1, 1, 8))
    cos, sin = rope_tables(1, 8)
    np.testing.assert_allclose(rope_fwd(x, cos, sin), x)


def test_no_position_table_and_odd_head_dim_rejected():
    m, _, _ = make()
    assert "wpe" not in m.params
    with pytest.raises(AssertionError):
        GPT(GPTConfig(11, 8, 1, 2, 6, rope=True))  # head dim 3


@pytest.mark.parametrize("extra", [{}, {"tie_weights": True}, {"dropout": 0.0}])
def test_grads_match_finite_differences(extra):
    m, idx, tgt = make(**extra)
    _, G = m.loss_and_grads(idx, tgt)
    assert "wpe" not in G
    rng = np.random.default_rng(0)
    for name in ["wte", "h0.qkv_W", "h1.qkv_W", "h0.qkv_b", "h1.proj_W", "lnf_g"]:
        for _ in range(8):
            ix = tuple(rng.integers(0, s) for s in m.params[name].shape)
            old = m.params[name][ix]
            m.params[name][ix] = old + 1e-6
            lp = m.forward(idx, tgt)[1]
            m.params[name][ix] = old - 1e-6
            lm = m.forward(idx, tgt)[1]
            m.params[name][ix] = old
            assert abs((lp - lm) / 2e-6 - G[name][ix]) < 1e-6, name


def test_grads_with_dropout_and_fixed_masks():
    m, idx, tgt = make(dropout=0.2)
    _, G = m.loss_and_grads(idx, tgt, rng=np.random.default_rng(5))
    for name in ["h0.qkv_W", "wte"]:
        ix = (1, 2)
        old = m.params[name][ix]
        vals = []
        for d in (1e-6, -1e-6):
            m.params[name][ix] = old + d
            vals.append(m.forward(idx, tgt, rng=np.random.default_rng(5))[1])
        m.params[name][ix] = old
        assert abs((vals[0] - vals[1]) / 2e-6 - G[name][ix]) < 1e-6


def test_causal():
    m, idx, _ = make()
    a = m.forward(idx)[0]
    idx2 = idx.copy()
    idx2[:, -1] = (idx2[:, -1] + 1) % 11
    b = m.forward(idx2)[0]
    np.testing.assert_allclose(a[:, :-1], b[:, :-1])


def test_runs_beyond_block_size_but_learned_positions_cannot():
    m, _, _ = make()
    x = np.random.default_rng(0).integers(0, 11, (1, 20))
    assert m.forward(x)[0].shape == (1, 20, 11)
    learned = GPT(GPTConfig(11, 8, 1, 2, 8))
    with pytest.raises(AssertionError):
        learned.forward(x)


def test_cached_next_token_matches_full_forward():
    m, idx, _ = make()
    full = m.forward(idx)[0][0, -1]
    out = generate_cached(m, [int(t) for t in idx[0]], 1, temperature=0.0001, seed=0)
    assert out[-1] == int(np.argmax(full))


@pytest.mark.parametrize("kw", [dict(), dict(top_k=4), dict(top_p=0.8)])
def test_kvcache_parity_including_sliding_window(kw):
    m, _, _ = make()
    prompt = [1, 2, 3]
    ref = m.generate(prompt, 20, seed=3, **kw)  # 23 tokens > block_size 8
    got = generate_cached(m, prompt, 20, seed=3, **kw)
    np.testing.assert_array_equal(ref, got)


def test_checkpoint_roundtrip(tmp_path):
    tok = CharTokenizer("abcdefghijk")
    m = GPT(GPTConfig(tok.vocab_size, 8, 1, 2, 8, rope=True), seed=3)
    p = str(tmp_path / "m.npz")
    save_checkpoint(p, m, tok)
    m2, _ = load_checkpoint(p)
    assert m2.cfg.rope and "wpe" not in m2.params
    x = np.array([[1, 2, 3]])
    np.testing.assert_allclose(m.forward(x)[0], m2.forward(x)[0])


def test_old_checkpoint_config_loads_as_learned_positions():
    cfg = GPTConfig(**{"vocab_size": 5, "block_size": 4, "n_layer": 1, "n_head": 1, "n_embd": 4})
    assert not cfg.rope and "wpe" in GPT(cfg).params
