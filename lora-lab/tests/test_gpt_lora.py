import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loralab.gpt_lora import GPTLoRA  # noqa: E402
from loralab.domain import generate_domain_text  # noqa: E402
from loralab import gpt_demo  # noqa: E402
from tinygpt import GPT, GPTConfig, Adam  # noqa: E402


def tiny(targets=("qkv", "proj"), r=2, seed=1):
    m = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=8), seed=seed, dtype=np.float64)
    rng = np.random.default_rng(2)
    for v in m.params.values():
        v += rng.standard_normal(v.shape) * 0.1
    lo = GPTLoRA(m, targets, r=r, alpha=4.0, seed=3)
    for n in lo.names:  # non-zero B so every gradient is non-trivial
        lo.B[n][...] = rng.standard_normal(lo.B[n].shape) * 0.3
    idx, tgt = rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))
    return m, lo, idx, tgt


def test_zero_init_matches_base_logits():
    m = GPT(GPTConfig(11, 8, 2, 2, 8), seed=0, dtype=np.float64)
    lo = GPTLoRA(m, r=2)
    idx = np.random.default_rng(0).integers(0, 11, (2, 6))
    assert np.allclose(lo.forward(idx)[0], m.forward(idx)[0])


def test_adapter_gradients_match_finite_differences():
    m, lo, idx, tgt = tiny(targets=("qkv", "proj", "fc", "out"))
    _, grads = lo.loss_and_grads(idx, tgt)
    rng = np.random.default_rng(4)
    eps = 1e-5
    for name, p in lo.params().items():
        for _ in range(3):
            j = tuple(rng.integers(0, s) for s in p.shape)
            old = p[j]
            p[j] = old + eps
            lp = lo.forward(idx, tgt)[1]
            p[j] = old - eps
            lm = lo.forward(idx, tgt)[1]
            p[j] = old
            assert abs((lp - lm) / (2 * eps) - grads[name][j]) < 1e-6 + 1e-4 * abs(lp - lm) / (2 * eps), name


def test_training_leaves_base_weights_untouched():
    m, lo, idx, tgt = tiny()
    before = {k: v.copy() for k, v in m.params.items()}
    opt = Adam(lo.params(), lr=1e-2)
    for _ in range(5):
        opt.step(lo.loss_and_grads(idx, tgt)[1])
    assert all(np.array_equal(before[k], m.params[k]) for k in before)


def test_training_reduces_loss():
    m, lo, idx, tgt = tiny()
    opt = Adam(lo.params(), lr=2e-2)
    first = lo.forward(idx, tgt)[1]
    for _ in range(40):
        opt.step(lo.loss_and_grads(idx, tgt)[1])
    assert lo.forward(idx, tgt)[1] < first - 0.1


def test_merge_equivalence_and_unmerge_restores():
    m, lo, idx, tgt = tiny()
    before = {k: v.copy() for k, v in m.params.items()}
    y = lo.forward(idx)[0]
    lo.merge()
    assert not np.allclose(m.forward(idx)[0], GPT(m.cfg, seed=1, dtype=np.float64).forward(idx)[0])
    assert np.allclose(m.forward(idx)[0], y)  # plain model now equals adapted model
    assert np.allclose(lo.forward(idx)[0], y)  # and the adapter does not double-apply
    lo.unmerge()
    assert all(np.allclose(before[k], m.params[k]) for k in before)
    with pytest.raises(AssertionError):
        lo.merge()
        lo.loss_and_grads(idx, tgt)


def test_disabling_adapter_gives_exact_base_model():
    m, lo, idx, _ = tiny()
    base_logits = m.forward(idx)[0]
    lo.enabled = False
    assert np.array_equal(lo.forward(idx)[0], base_logits)
    lo.enabled = True
    assert not np.allclose(lo.forward(idx)[0], base_logits)


def test_param_count_and_target_validation():
    m, lo, *_ = tiny(targets=("qkv",), r=2)
    # per layer: A (8x2) + B (2x24)
    assert lo.n_trainable() == 2 * (8 * 2 + 2 * 24)
    with pytest.raises(ValueError):
        GPTLoRA(m, ("bogus",))


def test_adapter_save_load_roundtrip(tmp_path):
    m, lo, idx, _ = tiny()
    y = lo.forward(idx)[0]
    p = tmp_path / "ad.npz"
    lo.save(str(p))
    fresh = GPTLoRA(m, ("qkv", "proj"), r=2, alpha=4.0, seed=99)
    assert not np.allclose(fresh.forward(idx)[0], y)
    fresh.load(str(p))
    assert np.allclose(fresh.forward(idx)[0], y)
    with pytest.raises(ValueError):
        GPTLoRA(m, ("qkv",), r=2).load(str(p))


def test_domain_text_is_deterministic():
    assert generate_domain_text(10, 1) == generate_domain_text(10, 1)
    assert generate_domain_text(10, 1) != generate_domain_text(10, 2)


def test_end_to_end_adaptation_helps_and_keeps_base():
    o = gpt_demo.run(rank=4, pre_steps=60, ad_steps=60, embd=32, layers=1, heads=2, block=24)
    assert o["lora_B"] < o["base_B"] - 0.5
    assert o["base_frozen"]
    assert abs(o["merged_B"] - o["lora_B"]) < 1e-3
    assert o["lora_params"] < o["full_params"] / 4
    assert "LoRA trains" in gpt_demo.format_report(o)
