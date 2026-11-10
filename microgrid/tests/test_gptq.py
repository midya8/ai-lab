import numpy as np
from microgrid import MiniGPT
from microgrid.gptq import collect_hessians, gptq_quantize, gptq_model
from microgrid.wquant import fake_quant, quantize_array


def test_gptq_lowers_output_error_vs_rtn():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 24)) @ rng.normal(size=(24, 24))  # correlated inputs
    W = rng.normal(size=(24, 10))
    H = X.T @ X
    e_rtn = np.linalg.norm(X @ (fake_quant(W, 3, 0) - W))
    e_gptq = np.linalg.norm(X @ (gptq_quantize(W, H, 3) - W))
    assert e_gptq < e_rtn


def test_values_stay_on_the_grid():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 12)); W = rng.normal(size=(12, 5))
    out = gptq_quantize(W, X.T @ X, 4)
    _, s = quantize_array(W, 4, 0)
    codes = out / s
    assert np.allclose(codes, np.rint(codes)) and np.abs(codes).max() <= 7


def test_identity_hessian_equals_rtn():
    W = np.random.default_rng(2).normal(size=(10, 4))
    # diagonal H: no cross-row correlation, so no error to push -> plain rounding
    assert np.allclose(gptq_quantize(W, np.eye(10), 4, damp=0.0), fake_quant(W, 4, 0))


def test_dead_input_rows_are_zeroed_safely():
    X = np.random.default_rng(3).normal(size=(50, 6)); X[:, 2] = 0
    W = np.random.default_rng(4).normal(size=(6, 3))
    out = gptq_quantize(W, X.T @ X, 4)
    assert np.all(np.isfinite(out)) and np.all(out[2] == 0)


def test_hessians_cover_every_linear_and_model_unchanged():
    m = MiniGPT(13, d=16, heads=4, layers=2, ctx=12, seed=0)
    win = np.random.default_rng(0).integers(0, 13, (4, 12))
    H = collect_hessians(m, win)
    assert len(H) == 2 * 4 + 1 and all(h.shape[0] == h.shape[1] for h in H.values())
    before = m.head.w.data.copy()
    q = gptq_model(m, win, 4)
    assert np.array_equal(m.head.w.data, before) and not np.array_equal(q.head.w.data, before)
    assert q(win).data.shape == (4, 12, 13)
