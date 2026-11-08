import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loralab.quant import NF4, QuantizedBase, dequantize, pack4, quantize, unpack4  # noqa: E402
from loralab.gpt_lora import GPTLoRA  # noqa: E402
from tinygpt import GPT, GPTConfig, Adam  # noqa: E402


def gaussian(shape=(64, 96), seed=0):
    return np.random.default_rng(seed).standard_normal(shape)


def rel_err(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(a)


def test_nf4_codebook_is_sorted_with_exact_zero_and_unit_ends():
    assert NF4.size == 16 and np.all(np.diff(NF4) > 0)
    assert NF4[0] == -1.0 and NF4[-1] == 1.0 and 0.0 in NF4


def test_int8_error_is_within_half_a_step_per_block():
    W = gaussian()
    q = quantize(W, "int8", block=32)
    err = np.abs(W - dequantize(q, np.float64)).reshape(-1)
    step = np.repeat(q.scales.astype(np.float64), 32)[:W.size]
    assert np.all(err <= step / 2 + 1e-6)


def test_error_ordering_on_gaussian_weights_int8_lt_nf4_lt_int4():
    W = gaussian((128, 128))
    e = {s: rel_err(W, dequantize(quantize(W, s, 64), np.float64)) for s in ("int8", "nf4", "int4")}
    assert e["int8"] < 0.01 < e["nf4"] < e["int4"]


def test_smaller_blocks_reduce_error():
    W = gaussian((64, 128))
    e = [rel_err(W, dequantize(quantize(W, "nf4", b), np.float64)) for b in (256, 64, 16)]
    assert e[0] > e[1] > e[2]


def test_outlier_only_hurts_its_own_block():
    W = gaussian((8, 64))
    W2 = W.copy()
    W2[0, 0] = 500.0
    a, b = (dequantize(quantize(x, "nf4", 64), np.float64) for x in (W, W2))
    assert np.array_equal(a[1:], b[1:])  # untouched blocks identical
    assert rel_err(W[0], b[0]) > 5 * rel_err(W[0], a[0])


def test_zero_blocks_and_padding_round_trip():
    W = np.zeros((3, 7))
    assert np.array_equal(dequantize(quantize(W, "nf4", 16), np.float64), W)
    X = gaussian((5, 13))  # 65 values: not a multiple of the block
    assert dequantize(quantize(X, "int8", 16)).shape == (5, 13)


def test_nf4_reproduces_its_own_levels_exactly():
    W = np.tile(NF4, 4).reshape(4, 16)  # every block has absmax 1
    assert np.allclose(dequantize(quantize(W, "nf4", 16), np.float64), W, atol=1e-7)


def test_pack_unpack_round_trip_odd_length():
    c = np.random.default_rng(1).integers(0, 16, 37).astype(np.uint8)
    assert np.array_equal(unpack4(pack4(c), 37), c)


def test_unknown_scheme_and_bad_block_rejected():
    with pytest.raises(ValueError):
        quantize(gaussian(), "fp8")
    with pytest.raises(ValueError):
        quantize(gaussian(), "nf4", block=0)


def small_model(seed=0):
    return GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16), seed=seed, dtype=np.float32)


def test_stored_bytes_ratio_by_scheme():
    ratios = {}
    for s in ("int8", "int4", "nf4"):
        qb = QuantizedBase(small_model(), s, block=64)
        ratios[s] = qb.compression()
    # fp32 -> 8 bit is just under 4x, 4 bit just under 8x (scales cost 4 bytes per 64 weights)
    assert 3.5 < ratios["int8"] < 4.0
    assert 6.0 < ratios["nf4"] < 8.0 and ratios["nf4"] == ratios["int4"]


def test_quantization_changes_only_targeted_tensors():
    m = small_model()
    before = {k: v.copy() for k, v in m.params.items()}
    qb = QuantizedBase(m, "nf4", block=32)
    for k, v in before.items():
        same = np.array_equal(v, m.params[k])
        assert same == (k not in qb.q), k
    assert "h0.qkv_W" in qb.q and "h0.qkv_b" not in qb.q and "wte" not in qb.q


def test_embeddings_flag_adds_embeddings():
    qb = QuantizedBase(small_model(), "int8", embeddings=True)
    assert {"wte", "wpe", "head_W"} <= set(qb.q)


def test_model_params_equal_dequantized_codes_and_relative_error_matches():
    m = small_model()
    orig = {k: v.copy() for k, v in m.params.items()}
    qb = QuantizedBase(m, "nf4", block=32)
    assert qb.verify_frozen()
    n = "h1.fc_W"
    assert abs(qb.tensor_errors()[n] - rel_err(orig[n].astype(np.float64), m.params[n].astype(np.float64))) < 1e-6


def test_lora_training_on_quantized_base_keeps_codes_intact_and_learns():
    m = small_model()
    rng = np.random.default_rng(3)
    idx, tgt = rng.integers(0, 11, (4, 8)), rng.integers(0, 11, (4, 8))
    qb = QuantizedBase(m, "nf4", block=32)
    lo = GPTLoRA(m, ("qkv", "proj"), r=2, seed=1)
    opt = Adam(lo.params(), lr=2e-2)
    first = lo.forward(idx, tgt)[1]
    for _ in range(40):
        opt.step(lo.loss_and_grads(idx, tgt)[1])
    assert lo.forward(idx, tgt)[1] < first - 0.05
    assert qb.verify_frozen()


def test_adapter_gradients_through_quantized_base_match_finite_differences():
    m = GPT(GPTConfig(11, 8, 2, 2, 8), seed=1, dtype=np.float64)
    QuantizedBase(m, "int4", block=16)
    lo = GPTLoRA(m, ("qkv", "fc"), r=2, alpha=4.0, seed=3)
    rng = np.random.default_rng(5)
    for n in lo.names:
        lo.B[n][...] = rng.standard_normal(lo.B[n].shape) * 0.3
    idx, tgt = rng.integers(0, 11, (2, 6)), rng.integers(0, 11, (2, 6))
    _, grads = lo.loss_and_grads(idx, tgt)
    eps = 1e-5
    for name, p in lo.params().items():
        j = tuple(rng.integers(0, s) for s in p.shape)
        old = p[j]
        p[j] = old + eps
        lp = lo.forward(idx, tgt)[1]
        p[j] = old - eps
        lm = lo.forward(idx, tgt)[1]
        p[j] = old
        assert abs((lp - lm) / (2 * eps) - grads[name][j]) < 1e-6 + 1e-4 * abs((lp - lm) / (2 * eps)), name


@pytest.mark.parametrize("scheme", ["int8", "int4", "nf4"])
def test_save_load_round_trip_is_exact(tmp_path, scheme):
    m = small_model()
    qb = QuantizedBase(m, scheme, block=32)
    path = tmp_path / "q.npz"
    qb.save(path)
    loaded = QuantizedBase.load_tensors(path)
    assert set(loaded) == set(qb.q)
    for n, q in loaded.items():
        assert np.array_equal(dequantize(q, np.float32), m.params[n]), n
    # packed 4-bit file is smaller than the fp32 weights it replaces
    assert os.path.getsize(path) < qb.fp_bytes()


def test_no_matching_targets_raises():
    with pytest.raises(ValueError):
        QuantizedBase(small_model(), "int8", targets=("nope",))


def test_qlora_demo_smoke_and_report_format():
    from loralab import qlora_demo
    rows = qlora_demo.run(rank=2, pre_steps=6, ad_steps=4, schemes=("fp32", "nf4"))
    assert [r["scheme"] for r in rows] == ["fp32", "nf4"]
    fp, nf = rows
    assert fp["weight_err"] == 0.0 and nf["weight_err"] > 0
    assert nf["model_bytes"] < fp["model_bytes"] / 3  # linear weights shrink ~7x, embeddings stay fp32
    assert all(r["base_intact"] for r in rows)
    assert fp["adapter_params"] == nf["adapter_params"]
    report = qlora_demo.format_report(rows)
    assert "nf4" in report and "untouched by training: True" in report
