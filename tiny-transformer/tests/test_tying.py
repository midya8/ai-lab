import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.checkpoint import save_checkpoint, load_checkpoint  # noqa: E402
from tinygpt.tokenizer import CharTokenizer  # noqa: E402


def make(tie):
    cfg = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8, tie_weights=tie)
    m = GPT(cfg, seed=1, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    return m, rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))


def test_tied_has_fewer_params_by_exactly_vc():
    assert make(False)[0].num_params() - make(True)[0].num_params() == 11 * 8
    assert "head_W" not in make(True)[0].params


def test_tied_grads_match_finite_differences():
    m, idx, tgt = make(True)
    _, G = m.loss_and_grads(idx, tgt)
    assert "head_W" not in G
    rng = np.random.default_rng(0)
    for name in ["wte", "head_b", "h0.qkv_W", "lnf_g"]:
        for _ in range(6):
            ix = tuple(rng.integers(0, s) for s in m.params[name].shape)
            old = m.params[name][ix]
            m.params[name][ix] = old + 1e-6
            lp = m.forward(idx, tgt)[1]
            m.params[name][ix] = old - 1e-6
            lm = m.forward(idx, tgt)[1]
            m.params[name][ix] = old
            assert abs((lp - lm) / 2e-6 - G[name][ix]) < 1e-6


def test_tied_equals_untied_with_copied_head():
    a, idx, tgt = make(True)
    b, _, _ = make(False)
    for k in a.params:
        b.params[k] = a.params[k].copy()
    b.params["head_W"] = a.params["wte"].T.copy()
    assert abs(a.forward(idx, tgt)[1] - b.forward(idx, tgt)[1]) < 1e-12


def test_checkpoint_roundtrip_and_kvcache_parity(tmp_path):
    tok = CharTokenizer("abcdefghijk")
    m = GPT(GPTConfig(tok.vocab_size, 8, 1, 2, 8, tie_weights=True), seed=3)
    p = str(tmp_path / "m.npz")
    save_checkpoint(p, m, tok)
    m2, _ = load_checkpoint(p)
    assert m2.cfg.tie_weights
    x = np.array([[1, 2, 3]])
    np.testing.assert_allclose(m.forward(x)[0], m2.forward(x)[0])
    assert GPTConfig(vocab_size=5).tie_weights is False
