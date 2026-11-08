import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loralab.gpt_lora import GPTLoRA  # noqa: E402
from loralab.loftq import _svd_adapter, loftq_init  # noqa: E402
from loralab.quant import QuantizedBase, dequantize, quantize  # noqa: E402
from tinygpt import GPT, Adam, GPTConfig  # noqa: E402

ALL = ("qkv", "proj", "fc", "out")


def make(seed=0, targets=ALL, r=4):
    m = GPT(GPTConfig(24, 16, 2, 2, 32), seed=seed)
    return m, GPTLoRA(m, targets, r=r, seed=seed)


def batch(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 24, (4, 16)), rng.integers(0, 24, (4, 16))


def test_svd_adapter_is_best_rank_r_approximation_of_the_residual():
    R = np.random.default_rng(0).standard_normal((12, 20))
    scale = 2.0
    A, B = _svd_adapter(R, 3, scale)
    approx = scale * A @ B
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    best = (U[:, :3] * S[:3]) @ Vt[:3]
    assert np.allclose(approx, best)
    assert np.linalg.matrix_rank(approx) == 3


def test_svd_adapter_pads_when_rank_exceeds_matrix_dimensions():
    R = np.random.default_rng(1).standard_normal((3, 5))
    A, B = _svd_adapter(R, 6, 1.0)
    assert A.shape == (3, 6) and B.shape == (6, 5)
    assert np.allclose(A @ B, R)  # rank >= min(dim): exact
    assert np.all(A[:, 3:] == 0) and np.all(B[3:] == 0)


@pytest.mark.parametrize("scheme", ["nf4", "int4", "int8"])
def test_one_iteration_never_worse_than_plain_quantization(scheme):
    m, lo = make()
    res = loftq_init(lo, scheme, block=32, iters=1)
    assert res.errors[0] <= res.baseline_error + 1e-9
    assert res.errors[0] < res.baseline_error  # and strictly better for a nonzero-rank adapter
    assert all(e <= 1 for e in res.tensor_errors.values())


def test_more_iterations_do_not_lose_to_plain_quantization_and_error_is_recorded_per_iter():
    m, lo = make()
    res = loftq_init(lo, "int4", block=32, iters=4)
    assert len(res.errors) == res.iters == 4
    assert res.final_error < res.baseline_error
    assert res.baseline_error == pytest.approx(QuantizedBase(make()[0], "int4", 32).relative_error(), rel=1e-6)


def test_reported_error_matches_an_independent_recomputation():
    m, lo = make()
    orig = {n: m.params[n].astype(np.float64) for n in lo.names}
    res = loftq_init(lo, "nf4", block=32, iters=3)
    num = sum(np.linalg.norm(orig[n] - m.params[n] - lo.delta(n)) ** 2 for n in lo.names)
    den = sum(np.linalg.norm(orig[n]) ** 2 for n in lo.names)
    assert np.sqrt(num / den) == pytest.approx(res.final_error, rel=1e-4)


def test_final_codes_are_the_quantization_of_w_minus_previous_delta():
    # with iters=2 the stored Q must be quantize(W - s*A1@B1), where (A1,B1) is the iteration-1 adapter
    m, lo = make()
    orig = {n: m.params[n].astype(np.float64) for n in lo.names}
    res = loftq_init(lo, "int4", block=32, iters=2)
    n = lo.names[0]
    q1 = quantize(orig[n], "int4", 32)
    A1, B1 = _svd_adapter(orig[n] - dequantize(q1, np.float64), lo.r, lo.scale)
    expect = quantize(orig[n] - lo.scale * A1 @ B1, "int4", 32)
    assert np.array_equal(res.qbase.q[n].codes, expect.codes)
    # and the final adapter is the SVD of that Q's residual
    A2, B2 = _svd_adapter(orig[n] - dequantize(expect, np.float64), lo.r, lo.scale)
    assert np.allclose(lo.scale * lo.A[n] @ lo.B[n], lo.scale * A2 @ B2, atol=1e-5)


def test_untargeted_tensors_are_quantized_plainly():
    m, lo = make(targets=("qkv", "proj"))
    ref, _ = make(targets=("qkv", "proj"))
    loftq_init(lo, "nf4", block=32, iters=2)
    want = QuantizedBase(ref, "nf4", 32)
    for n in want.names:
        if n not in lo.names:
            assert np.array_equal(m.params[n], ref.params[n])


def test_adapted_model_starts_closer_to_fp32_than_the_quantized_base():
    m, lo = make()
    x, y = batch()
    ref_logits = m.forward(x)[0].copy()
    res = loftq_init(lo, "int4", block=16, iters=3)
    with_adapter = lo.forward(x)[0]
    with lo.active(False) as base_only:
        no_adapter = base_only.forward(x)[0]
    assert np.abs(with_adapter - ref_logits).mean() < np.abs(no_adapter - ref_logits).mean()
    assert res.qbase.verify_frozen()


def test_gradients_still_match_finite_differences_from_a_loftq_start():
    m, lo = make(r=2)
    loftq_init(lo, "nf4", block=16, iters=2)
    x, y = batch()
    _, g = lo.loss_and_grads(x, y)
    n = lo.names[0]
    rng = np.random.default_rng(3)
    for key, arr in ((f"{n}.A", lo.A[n]), (f"{n}.B", lo.B[n])):
        for _ in range(3):
            i = tuple(rng.integers(0, s) for s in arr.shape)
            old = arr[i]
            eps = 1e-3
            arr[i] = old + eps
            hi = lo.loss_and_grads(x, y)[0]
            arr[i] = old - eps
            lo_ = lo.loss_and_grads(x, y)[0]
            arr[i] = old
            assert g[key][i] == pytest.approx((hi - lo_) / (2 * eps), rel=0.05, abs=2e-4)


def test_base_stays_frozen_after_training_from_a_loftq_start():
    m, lo = make()
    res = loftq_init(lo, "int4", block=32, iters=2)
    opt = Adam(lo.params(), lr=1e-2)
    x, y = batch()
    losses = []
    for _ in range(15):
        loss, g = lo.loss_and_grads(x, y)
        losses.append(loss)
        opt.step(g)
    assert losses[-1] < losses[0]
    assert res.qbase.verify_frozen()


def test_merge_matches_adapted_forward():
    m, lo = make()
    loftq_init(lo, "nf4", block=32, iters=2)
    x, y = batch()
    before = lo.forward(x)[0].copy()
    lo.merge()
    assert np.allclose(m.forward(x)[0], before, atol=1e-4)


def test_rejects_adapters_on_tensors_that_are_not_quantized():
    _, lo = make(targets=("qkv", "proj"))
    with pytest.raises(ValueError):
        loftq_init(lo, "nf4", targets=("qkv_W",))


def test_rejects_zero_iterations():
    _, lo = make()
    with pytest.raises(ValueError):
        loftq_init(lo, "nf4", iters=0)


def test_demo_smoke():
    from loralab.loftq_demo import format_report, run
    fp, rows = run(rank=2, iters=2, pre_steps=5, ad_steps=3, schemes=("int4",))
    assert [r["init"] for r in rows] == ["qlora", "svd", "loftq"]
    assert rows[1]["weight_err"] < rows[0]["weight_err"]
    assert "loftq" in format_report(fp, rows)
