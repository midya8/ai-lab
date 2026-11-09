import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loralab.quant import QuantizedBase, _dq_decode, _dq_encode, dequantize, quantize  # noqa: E402
from loralab.gpt_lora import GPTLoRA  # noqa: E402
from loralab import dq_frontier  # noqa: E402
from tinygpt import GPT, GPTConfig, Adam  # noqa: E402


def gaussian(shape=(96, 128), seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def small_model(seed=0, dtype=np.float32):
    return GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16), seed=seed, dtype=dtype)


def test_scale_codec_error_is_within_half_a_second_level_step():
    s = np.random.default_rng(1).uniform(0.5, 4.0, 1000)
    dq = _dq_encode(s, 256)
    err = np.abs(_dq_decode(dq).astype(np.float64) - s)
    step = np.repeat(dq["scales"].astype(np.float64), 256)[:s.size]
    assert np.all(err <= step / 2 + 1e-6)
    assert dq["codes"].dtype == np.int8 and dq["codes"].size == s.size  # padding is not stored


def test_mean_centering_matters_for_scales_far_from_zero():
    s = 100.0 + np.random.default_rng(2).uniform(0, 1, 512)  # tight cluster far from 0
    dq = _dq_encode(s, 256)
    assert np.abs(_dq_decode(dq) - s).max() < 0.01  # without the mean the step would be ~100/127 = 0.8


def test_decoded_scales_never_reach_zero_even_with_extreme_outlier_blocks():
    s = np.concatenate([np.full(255, 1e-6), [1e6]])
    assert _dq_decode(_dq_encode(s, 256)).min() > 0


@pytest.mark.parametrize("scheme", ["int8", "int4", "nf4"])
def test_double_quant_changes_error_only_slightly_on_gaussian_weights(scheme):
    W = gaussian((128, 128))
    plain = rel_err(W, dequantize(quantize(W, scheme, 64), np.float64))
    dq = rel_err(W, dequantize(quantize(W, scheme, 64, double_quant=True), np.float64))
    assert abs(dq - plain) / plain < 0.05


def test_codes_are_chosen_against_the_dequantized_scales():
    # re-quantizing the dequantized tensor with the *same effective scales* must reproduce the codes
    W = gaussian((64, 64), seed=3)
    q = quantize(W, "int4", 32, double_quant=True)
    B = W.reshape(-1, 32)
    manual = np.clip(np.rint(B / q.scales.astype(np.float64)[:, None]), -7, 7).astype(np.int8)
    assert np.array_equal(manual, q.codes)


def test_bits_per_weight_matches_the_qlora_paper_numbers():
    W = gaussian((512, 512))  # 4096 blocks of 64 -> 16 second-level blocks
    plain = quantize(W, "nf4", 64)
    dq = quantize(W, "nf4", 64, double_quant=True)
    assert plain.bits_per_weight == pytest.approx(4.5)
    assert dq.bits_per_weight == pytest.approx(4 + 8 / 64 + 32 / (64 * 256), abs=1e-3)  # ~4.127
    assert dq.stored_bytes < plain.stored_bytes


def test_dq_block_size_controls_second_level_overhead():
    W = gaussian((256, 256))
    small = quantize(W, "nf4", 16, double_quant=True, dq_block=32)
    large = quantize(W, "nf4", 16, double_quant=True, dq_block=1024)
    assert small.stored_bytes > large.stored_bytes


def test_zero_tensor_and_padding_round_trip_with_double_quant():
    assert np.array_equal(dequantize(quantize(np.zeros((3, 7)), "nf4", 16, double_quant=True), np.float64), np.zeros((3, 7)))
    X = gaussian((5, 13))
    assert dequantize(quantize(X, "int8", 16, double_quant=True)).shape == (5, 13)


def test_invalid_dq_block_rejected():
    with pytest.raises(ValueError):
        quantize(gaussian(), "nf4", 64, double_quant=True, dq_block=0)


def test_base_reports_fewer_bits_and_keeps_params_equal_to_dequantized_codes():
    plain = QuantizedBase(small_model(), "nf4", block=16)
    m = small_model()
    dq = QuantizedBase(m, "nf4", block=16, double_quant=True, dq_block=32)
    assert dq.bits_per_weight() < plain.bits_per_weight() - 1.0  # 6.0 -> ~4.5
    assert dq.verify_frozen() and dq.stored_bytes() < plain.stored_bytes()


@pytest.mark.parametrize("scheme", ["int8", "int4", "nf4"])
def test_save_load_round_trip_is_bit_exact_with_double_quant(tmp_path, scheme):
    m = small_model()
    qb = QuantizedBase(m, scheme, block=16, double_quant=True, dq_block=64)
    path = tmp_path / "dq.npz"
    qb.save(path)
    loaded = QuantizedBase.load_tensors(path)
    for n, q in loaded.items():
        assert q.dq is not None and np.array_equal(q.scales, qb.q[n].scales), n
        assert np.array_equal(dequantize(q, np.float32), m.params[n]), n
        assert q.stored_bytes == qb.q[n].stored_bytes


def test_double_quant_file_is_smaller_on_a_realistic_tensor_and_old_files_still_load(tmp_path):
    # (a toy model's many tiny tensors are dominated by .npz per-array overhead, so measure one big tensor)
    big = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=128), seed=0, dtype=np.float32)
    pa, pb = tmp_path / "big_plain.npz", tmp_path / "big_dq.npz"
    QuantizedBase(big, "nf4", block=16, targets=("fc_W",)).save(pa)
    big2 = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=2, n_embd=128), seed=0, dtype=np.float32)
    QuantizedBase(big2, "nf4", block=16, targets=("fc_W",), double_quant=True).save(pb)
    assert os.path.getsize(pb) < 0.85 * os.path.getsize(pa)
    a = tmp_path / "plain.npz"
    QuantizedBase(small_model(), "nf4", block=16).save(a)
    # a two-field meta (written before double quantization existed) loads as plain
    with np.load(a) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["__meta__"] = arrays["__meta__"][:2]
    old = tmp_path / "old.npz"
    np.savez(old, **arrays)
    assert all(q.dq is None for q in QuantizedBase.load_tensors(old).values())


def test_lora_trains_on_double_quantized_base_and_codes_stay_frozen():
    m = small_model()
    rng = np.random.default_rng(3)
    idx, tgt = rng.integers(0, 11, (4, 8)), rng.integers(0, 11, (4, 8))
    qb = QuantizedBase(m, "nf4", block=16, double_quant=True, dq_block=32)
    lo = GPTLoRA(m, ("qkv", "proj"), r=2, seed=1)
    opt = Adam(lo.params(), lr=2e-2)
    first = lo.forward(idx, tgt)[1]
    for _ in range(40):
        opt.step(lo.loss_and_grads(idx, tgt)[1])
    assert lo.forward(idx, tgt)[1] < first - 0.05
    assert qb.verify_frozen()


def test_pareto_helper_marks_non_dominated_points_only():
    pts = [(4.0, 0.30), (5.0, 0.20), (5.0, 0.25), (6.0, 0.10), (4.5, 0.35)]
    assert dq_frontier.pareto(pts) == [0, 1, 3]
    assert dq_frontier.pareto([(1.0, 1.0), (1.0, 1.0)]) == [0, 1]  # ties are both kept


def test_frontier_smoke_and_small_block_with_dq_beats_big_plain_block_at_same_bits():
    rows, fp = dq_frontier.run(seeds=1, schemes=("nf4",), blocks=(16, 64), pre_steps=4, block_size=16)
    by = {(r["block"], r["double_quant"]): r for r in rows}
    assert len(rows) == 4 and fp > 0
    assert by[(16, True)]["bits"] < by[(64, False)]["bits"] + 0.05  # ~4.5 vs 4.5
    assert by[(16, True)]["err"] < by[(64, False)]["err"]
    assert by[(64, True)]["bits"] < by[(64, False)]["bits"]
    assert "pareto" in dq_frontier.format_report(rows, fp)


def test_qlora_demo_accepts_double_quant():
    from loralab import qlora_demo
    rows = qlora_demo.run(rank=2, pre_steps=4, ad_steps=3, schemes=("nf4",), double_quant=True)
    assert rows[0]["base_intact"]
