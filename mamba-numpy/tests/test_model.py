import numpy as np
import pytest

from mamba import Mamba, SSM, selective_copy, associative_recall
from mamba.layers import CausalDWConv, RMSNorm, silu, silu_backward


def fd_check(loss_fn, params, grads, rng, n_probe=6, eps=1e-6, rtol=2e-4, atol=1e-7):
    """Central finite differences on random entries of every parameter."""
    bad = []
    for name, p in params.items():
        flat = p.reshape(-1)
        idxs = rng.choice(flat.size, min(n_probe, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps; lp = loss_fn()
            flat[i] = old - eps; lm = loss_fn()
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            if abs(num - ana) > atol + rtol * max(abs(num), abs(ana)):
                bad.append((name, int(i), num, ana))
    return bad


@pytest.mark.parametrize("selective", [True, False])
def test_ssm_layer_gradients(selective):
    rng = np.random.default_rng(1)
    ssm = SSM(5, 3, rng, selective=selective)
    u = rng.normal(size=(2, 7, 5))
    w = rng.normal(size=(2, 7, 5))
    loss = lambda: float((ssm.forward(u) * w).sum())
    loss()
    du = ssm.backward(w)
    assert not fd_check(loss, ssm.p, ssm.g, rng, n_probe=8)
    # input gradient
    bad = fd_check(loss, {"u": u}, {"u": du}, rng, n_probe=20)
    assert not bad


def test_ssm_parallel_scan_gives_same_forward_and_grads():
    rng = np.random.default_rng(2)
    a = SSM(4, 3, np.random.default_rng(5), scan_method="sequential")
    b = SSM(4, 3, np.random.default_rng(5), scan_method="parallel")
    u = rng.normal(size=(2, 11, 4))
    w = rng.normal(size=(2, 11, 4))
    np.testing.assert_allclose(a.forward(u), b.forward(u), atol=1e-11)
    np.testing.assert_allclose(a.backward(w), b.backward(w), atol=1e-10)
    for k in a.g:
        np.testing.assert_allclose(a.g[k], b.g[k], atol=1e-9)


def test_small_layers_gradients():
    rng = np.random.default_rng(3)
    for layer, shape in [(RMSNorm(6), (2, 4, 6)), (CausalDWConv(5, 3, rng), (2, 6, 5))]:
        x = rng.normal(size=shape)
        w = rng.normal(size=shape)
        loss = lambda: float((layer.forward(x) * w).sum())
        loss()
        dx = layer.backward(w)
        assert not fd_check(loss, layer.p, layer.g, rng)
        assert not fd_check(loss, {"x": x}, {"x": dx}, rng, n_probe=15)
    x = rng.normal(size=(30,))
    num = (silu(x + 1e-6) - silu(x - 1e-6)) / 2e-6
    np.testing.assert_allclose(silu_backward(np.ones_like(x), x), num, atol=1e-8)


@pytest.mark.parametrize("selective", [True, False])
def test_full_model_gradients(selective):
    m = Mamba(vocab=9, d_model=6, d_state=3, n_layer=2, selective=selective, seed=0)
    rng = np.random.default_rng(0)
    toks, tgts = selective_copy(3, 6, 2, 7, rng)
    loss = lambda: m.loss_and_grads(toks, tgts)[0]
    loss()
    grads = {k: v.copy() for k, v in m.grads().items()}
    assert set(grads) == set(m.params())
    bad = fd_check(loss, m.params(), grads, rng, n_probe=4)
    assert not bad, bad[:3]


def test_gradient_check_catches_a_broken_backward(monkeypatch):
    """The finite-difference harness must fail if the SSM backward is wrong (here: drop the A-path)."""
    m = Mamba(vocab=9, d_model=6, d_state=3, n_layer=1, seed=0)
    orig = SSM.backward

    def broken(self, dy):
        du = orig(self, dy)
        self.g["A_log"] = self.g["A_log"] * 0.5
        return du

    monkeypatch.setattr(SSM, "backward", broken)
    rng = np.random.default_rng(0)
    toks, tgts = selective_copy(3, 6, 2, 7, rng)
    loss = lambda: m.loss_and_grads(toks, tgts)[0]
    loss()
    grads = {k: v.copy() for k, v in m.grads().items()}
    assert fd_check(loss, m.params(), grads, rng, n_probe=6)


def test_masked_positions_get_no_gradient():
    m = Mamba(vocab=9, d_model=6, d_state=3, n_layer=1, seed=0)
    toks = np.array([[1, 2, 3, 4]])
    tg = np.array([[-1, -1, -1, -1]])
    loss, _ = m.loss_and_grads(toks, tg)
    assert loss == 0.0
    assert all(np.abs(g).max() == 0 for g in m.grads().values())


def test_causality():
    m = Mamba(vocab=9, d_model=6, d_state=3, n_layer=2, seed=1)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 9, (2, 10))
    b = a.copy()
    b[:, 6:] = rng.integers(0, 9, (2, 4))
    la, lb = m.forward(a), m.forward(b)
    np.testing.assert_allclose(la[:, :6], lb[:, :6], atol=1e-12)
    assert np.abs(la[:, 6:] - lb[:, 6:]).max() > 1e-6


@pytest.mark.parametrize("selective", [True, False])
def test_step_decoding_matches_full_forward(selective):
    m = Mamba(vocab=9, d_model=6, d_state=3, n_layer=2, selective=selective, seed=2)
    toks = np.random.default_rng(0).integers(0, 9, (3, 12))
    full = m.forward(toks)
    state = m.init_state(3)
    for t in range(12):
        lg, state = m.step(toks[:, t], state)
        np.testing.assert_allclose(lg, full[:, t], atol=1e-10)


def test_generate_is_deterministic_and_matches_argmax_of_forward():
    m = Mamba(vocab=9, d_model=6, d_state=3, n_layer=1, seed=3)
    out = m.generate([1, 2, 3], 5)
    assert out == m.generate([1, 2, 3], 5)
    for t in range(3, 8):
        assert out[t] == int(np.argmax(m.forward(np.array([out[:t]]))[0, -1]))


def test_lti_layer_is_a_convolution_with_the_closed_form_kernel():
    rng = np.random.default_rng(0)
    ssm = SSM(4, 3, rng, selective=False)
    u = rng.normal(size=(2, 9, 4))
    K = ssm.kernel(9)  # (T, D)
    y = np.zeros_like(u)
    for t in range(9):
        for s in range(t + 1):
            y[:, t] += K[t - s] * u[:, s]
    y += ssm.p["Dskip"] * u
    np.testing.assert_allclose(ssm.forward(u), y, atol=1e-11)


def test_lti_is_linear_and_selective_is_not():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 1, 8, 5))
    lti = SSM(5, 3, np.random.default_rng(1), selective=False)
    sel = SSM(5, 3, np.random.default_rng(1), selective=True)
    gap = lambda l: np.abs(l.forward(a + b) - l.forward(a) - l.forward(b)).max()
    assert gap(lti) < 1e-12
    assert gap(sel) > 1e-3
    with pytest.raises(ValueError):
        sel.kernel(4)


def test_state_size_is_independent_of_length():
    m = Mamba(vocab=9, d_model=8, d_state=4, n_layer=2)
    st = m.init_state(1)
    total = sum(s["h"].nbytes + s["buf"].nbytes for s in st)
    assert total == m.state_bytes()
    for _ in range(50):
        _, st = m.step(np.array([1]), st)
    assert sum(s["h"].nbytes + s["buf"].nbytes for s in st) == total


def test_tasks_shapes_and_labels():
    rng = np.random.default_rng(0)
    t, y = selective_copy(5, 12, 3, 6, rng)
    assert t.shape == y.shape == (5, 15)
    for i in range(5):
        data = t[i, :12][t[i, :12] > 0]
        np.testing.assert_array_equal(data, y[i, 12:])
        assert (y[i, :12] == -1).all()
    t, y = selective_copy(2, 12, 3, 6, rng, fixed_slots=True)
    assert (t[:, :12] > 0).sum(1).tolist() == [3, 3]
    t, y = associative_recall(6, 4, 8, 5, rng)
    assert t.shape == (6, 10)
    for i in range(6):
        q = t[i, 8]
        j = list(t[i, 0:8:2]).index(q)
        assert y[i, 8] == t[i, 2 * j + 1]
