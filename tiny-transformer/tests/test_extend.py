import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, generate_cached  # noqa: E402
from tinygpt.checkpoint import evaluate, load_checkpoint, save_checkpoint  # noqa: E402
from tinygpt.cli import main as cli_main  # noqa: E402
from tinygpt.extend import extend_context, finetune_extended, with_window  # noqa: E402
from tinygpt.model import rope_inv_freq, rope_tables  # noqa: E402
from tinygpt.tokenizer import CharTokenizer  # noqa: E402


def make(block=8, **kw):
    cfg = GPTConfig(vocab_size=11, block_size=block, n_layer=2, n_head=2, n_embd=16, rope=True, **kw)
    m = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    return m


# ---- the tables ----
def test_scale_one_is_plain_rope_for_both_modes():
    ref = rope_tables(12, 8)
    for mode in ("linear", "ntk"):
        c, s = rope_tables(12, 8, scale=1.0, mode=mode)
        np.testing.assert_array_equal(c, ref[0])
        np.testing.assert_array_equal(s, ref[1])


def test_linear_scale_maps_position_p_to_p_over_scale():
    c, s = rope_tables(1, 8, offset=10, scale=2.5, mode="linear")
    c0, s0 = rope_tables(1, 8, offset=4)
    np.testing.assert_allclose(c, c0)
    np.testing.assert_allclose(s, s0)


def test_ntk_stretches_slowest_pair_by_scale_and_barely_moves_fastest():
    D, scale = 16, 4.0
    plain = rope_inv_freq(D)
    ntk = rope_inv_freq(D, scale=scale, mode="ntk")
    assert ntk[0] == pytest.approx(plain[0])  # highest frequency untouched
    assert plain[-1] / ntk[-1] == pytest.approx(scale)  # lowest frequency stretched exactly `scale`x
    ratios = plain / ntk
    assert np.all(np.diff(ratios) > 0)  # stretch grows monotonically towards low frequencies


def test_invalid_scaling_arguments_rejected():
    with pytest.raises(ValueError):
        rope_tables(4, 8, mode="yarn")
    with pytest.raises(ValueError):
        rope_tables(4, 8, scale=0)
    with pytest.raises(AssertionError):
        GPT(GPTConfig(11, 8, 1, 2, 8, rope_scale=2.0))  # scale without rope
    with pytest.raises(AssertionError):
        GPT(GPTConfig(11, 8, 1, 2, 8, rope=True, rope_scaling="bogus"))


def test_relative_position_property_holds_under_scaling():
    rng = np.random.default_rng(0)
    q, k = rng.standard_normal(8), rng.standard_normal(8)
    from tinygpt.model import rope_fwd

    def score(m, n, **kw):
        cm, sm = rope_tables(1, 8, offset=m, **kw)
        cn, sn = rope_tables(1, 8, offset=n, **kw)
        return float(rope_fwd(q[None], cm, sm)[0] @ rope_fwd(k[None], cn, sn)[0])

    for kw in (dict(scale=3.0, mode="linear"), dict(scale=3.0, mode="ntk")):
        assert score(40, 37, **kw) == pytest.approx(score(140, 137, **kw))


# ---- extend_context ----
def test_view_shares_weights_and_records_provenance():
    m = make()
    v = extend_context(m, 24, "linear")
    assert v.params is m.params
    assert (v.cfg.block_size, v.cfg.rope_scale, v.cfg.rope_scaling, v.cfg.train_len) == (24, 3.0, "linear", 8)
    assert m.cfg.block_size == 8 and m.cfg.rope_scale == 1.0  # original untouched


def test_extension_compounds_against_the_trained_length_not_the_last_view():
    m = make()
    v2 = extend_context(extend_context(m, 16, "linear"), 32, "linear")
    assert v2.cfg.rope_scale == 4.0 and v2.cfg.train_len == 8
    back = extend_context(v2, 8, "linear")
    assert back.cfg.rope_scale == 1.0
    x = np.random.default_rng(0).integers(0, 11, (2, 8))
    np.testing.assert_allclose(back.forward(x)[0], m.forward(x)[0])


def test_none_mode_is_plain_extrapolation_and_identical_inside_the_window():
    m = make()
    v = extend_context(m, 24, "none")
    assert v.cfg.rope_scale == 1.0
    x = np.random.default_rng(0).integers(0, 11, (2, 8))
    np.testing.assert_allclose(v.forward(x)[0], m.forward(x)[0])


@pytest.mark.parametrize("mode", ["linear", "ntk"])
def test_scaling_changes_the_function_but_stays_causal(mode):
    m = make()
    v = extend_context(m, 24, mode)
    x = np.random.default_rng(0).integers(0, 11, (1, 12))
    a = v.forward(x)[0]
    assert not np.allclose(a, m.forward(x)[0])
    x2 = x.copy()
    x2[:, -1] = (x2[:, -1] + 1) % 11
    np.testing.assert_allclose(a[:, :-1], v.forward(x2)[0][:, :-1])


def test_learned_position_model_cannot_be_extended():
    m = GPT(GPTConfig(11, 8, 1, 2, 8))
    with pytest.raises(ValueError):
        extend_context(m, 16)


def test_bad_arguments():
    m = make()
    with pytest.raises(ValueError):
        extend_context(m, 16, "yarn")
    with pytest.raises(ValueError):
        extend_context(m, 0)


@pytest.mark.parametrize("mode", ["linear", "ntk"])
def test_grads_match_finite_differences_under_scaling(mode):
    v = extend_context(make(), 24, mode)
    rng = np.random.default_rng(3)
    idx, tgt = rng.integers(0, 11, (2, 12)), rng.integers(0, 11, (2, 12))
    _, G = v.loss_and_grads(idx, tgt)
    for name in ["wte", "h0.qkv_W", "h1.qkv_W", "h1.proj_W"]:
        for _ in range(8):
            ix = tuple(rng.integers(0, s) for s in v.params[name].shape)
            old = v.params[name][ix]
            v.params[name][ix] = old + 1e-6
            lp = v.forward(idx, tgt)[1]
            v.params[name][ix] = old - 1e-6
            lm = v.forward(idx, tgt)[1]
            v.params[name][ix] = old
            assert abs((lp - lm) / 2e-6 - G[name][ix]) < 1e-6, name


@pytest.mark.parametrize("mode", ["linear", "ntk"])
def test_kv_cache_parity_on_extended_model(mode):
    v = extend_context(make(), 20, mode)
    prompt = [1, 2, 3]
    ref = v.generate(prompt, 30, seed=4, top_k=5)  # 33 tokens > extended window 20
    got = generate_cached(v, prompt, 30, seed=4, top_k=5)
    np.testing.assert_array_equal(ref, got)


def test_extended_model_works_with_gqa():
    cfg = GPTConfig(11, 8, 1, 4, 16, rope=True, n_kv_head=2)
    v = extend_context(GPT(cfg, seed=0, dtype=np.float64), 24, "ntk")
    x = np.random.default_rng(0).integers(0, 11, (1, 20))
    np.testing.assert_allclose(v.forward(x)[0][0, -1], generate_logits_cached(v, x[0]), atol=1e-9)


def generate_logits_cached(model, ids):
    from tinygpt.kvcache import _prefill
    return _prefill(model, [int(t) for t in ids])[0]


# ---- persistence ----
def test_checkpoint_roundtrip_keeps_scaling(tmp_path):
    tok = CharTokenizer("abcdefghijk")
    m = GPT(GPTConfig(tok.vocab_size, 8, 1, 2, 8, rope=True), seed=3)
    v = extend_context(m, 20, "ntk")
    p = str(tmp_path / "v.npz")
    save_checkpoint(p, v, tok)
    v2, _ = load_checkpoint(p)
    assert (v2.cfg.block_size, v2.cfg.rope_scaling, v2.cfg.train_len) == (20, "ntk", 8)
    assert v2.cfg.rope_scale == pytest.approx(2.5)
    x = np.random.default_rng(0).integers(0, tok.vocab_size, (1, 15))
    np.testing.assert_allclose(v.forward(x)[0], v2.forward(x)[0], atol=1e-6)


def test_pre_extension_checkpoint_meta_still_loads(tmp_path):
    tok = CharTokenizer("abcdefghijk")
    m = GPT(GPTConfig(tok.vocab_size, 8, 1, 2, 8, rope=True), seed=3)
    p = str(tmp_path / "old.npz")
    save_checkpoint(p, m, tok)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(str(arrays.pop("__meta__")))
    for k in ("rope_scale", "rope_scaling", "train_len"):
        meta["config"].pop(k)
    np.savez(p, __meta__=np.array(json.dumps(meta)), **arrays)
    m2, _ = load_checkpoint(p)
    assert m2.cfg.rope_scale == 1.0 and m2.cfg.train_len == 0


# ---- fine-tuning ----
def test_finetune_updates_the_shared_weights_and_lowers_loss():
    m = make()
    v = extend_context(m, 16, "linear")
    data = np.tile(np.arange(11), 30)
    before = m.params["wte"].copy()
    losses = finetune_extended(v, data, steps=40, batch=4, lr=3e-3)
    assert len(losses) == 40 and losses[-1] < losses[0]
    assert not np.allclose(before, m.params["wte"])  # the original sees the update
    with pytest.raises(ValueError):
        finetune_extended(v, np.arange(10), steps=1)


def test_evaluate_on_extended_window_scores_more_tokens_per_window():
    m = make()
    data = np.random.default_rng(0).integers(0, 11, 200)
    r = evaluate(extend_context(m, 32, "linear"), data)
    assert r["tokens"] == (199 // 32) * 32 and np.isfinite(r["loss"])


# ---- CLI ----
def test_cli_extend_end_to_end(tmp_path, capsys):
    text = "the quick brown fox jumps over the lazy dog. " * 40
    tok = CharTokenizer(text)
    m = GPT(GPTConfig(tok.vocab_size, 16, 1, 2, 16, rope=True), seed=0)
    src, dst, data = str(tmp_path / "a.npz"), str(tmp_path / "b.npz"), str(tmp_path / "t.txt")
    save_checkpoint(src, m, tok)
    open(data, "w").write(text)
    cli_main(["extend", src, "--len", "48", "--mode", "linear", "--data", data,
              "--finetune-steps", "5", "--batch", "2", "--out", dst])
    out = capsys.readouterr().out
    assert "fine-tuned 5 steps" in out and "block_size=48" in out
    m2, _ = load_checkpoint(dst)
    assert m2.cfg.block_size == 48 and m2.cfg.rope_scale == 3.0
    cli_main(["eval", dst, "--data", data])
    assert "perplexity" in capsys.readouterr().out


def test_with_window_keeps_scaling_and_weights_but_changes_window():
    m = make()
    v = extend_context(m, 24, "linear")
    w = with_window(v, 8)
    assert w.params is m.params
    assert (w.cfg.block_size, w.cfg.rope_scale, w.cfg.rope_scaling, w.cfg.train_len) == (8, 3.0, "linear", 8)
    x = np.random.default_rng(0).integers(0, 11, (1, 8))
    np.testing.assert_allclose(w.forward(x)[0], v.forward(x)[0])  # same function on the same tokens
    assert not np.allclose(w.forward(x)[0], m.forward(x)[0])  # but not the unscaled model
    with pytest.raises(ValueError):
        with_window(v, 0)
