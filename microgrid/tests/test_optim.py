import numpy as np
from microgrid import Tensor, Adam, AdamW, clip_grad_norm, cosine_warmup


def _param(v):
    t = Tensor(np.array(v, dtype=float))
    t.grad = np.zeros_like(t.data)
    return t


def test_adamw_zero_grad_decays_weights_exactly():
    p = _param([1.0, -2.0])
    AdamW([p], lr=0.1, weight_decay=0.5).step()
    np.testing.assert_allclose(p.data, np.array([1.0, -2.0]) * (1 - 0.1 * 0.5))


def test_adamw_wd0_matches_adam():
    a, b = _param([1.0, 2.0]), _param([1.0, 2.0])
    oa, ob = Adam([a], lr=0.05), AdamW([b], lr=0.05, weight_decay=0.0)
    for g in ([0.3, -0.1], [0.2, 0.4]):
        a.grad[:] = g; b.grad[:] = g
        oa.step(); ob.step()
    np.testing.assert_allclose(a.data, b.data)


def test_clip_grad_norm():
    p, q = _param([0.0, 0.0]), _param([0.0])
    p.grad[:] = [3.0, 0.0]; q.grad[:] = [4.0]
    n = clip_grad_norm([p, q], 1.0)
    assert abs(n - 5.0) < 1e-12
    assert abs(np.sqrt((p.grad ** 2).sum() + (q.grad ** 2).sum()) - 1.0) < 1e-6
    p.grad[:] = [0.1, 0.0]; q.grad[:] = [0.1]
    before = p.grad.copy()
    clip_grad_norm([p, q], 1.0)
    np.testing.assert_array_equal(p.grad, before)


def test_cosine_warmup_shape():
    assert abs(cosine_warmup(5, 100, 1.0, warmup=10) - 0.5) < 1e-12
    assert abs(cosine_warmup(10, 100, 1.0, warmup=10) - 1.0) < 1e-12
    assert abs(cosine_warmup(100, 100, 1.0, warmup=10, min_lr=0.1) - 0.1) < 1e-12
    lrs = [cosine_warmup(s, 100, 1.0, 10) for s in range(10, 101)]
    assert all(x >= y - 1e-12 for x, y in zip(lrs, lrs[1:]))
