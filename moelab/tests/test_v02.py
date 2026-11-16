"""v0.2: shared experts and auxiliary-loss-free (bias) balancing."""
import numpy as np
import pytest

from moelab import MoELayer, Net, fit, load_stats
from test_moelab import _fd_check


def _net(n_shared=0, bias_rate=0.0, k=2, cf=1.25, E=4, d=6, seed=0, decisive=True, **kw):
    layer = MoELayer(d, 8, n_experts=E, k=k, capacity_factor=cf, n_shared=n_shared,
                     bias_rate=bias_rate, seed=seed, **kw)
    if decisive:
        layer.Wg = layer.Wg * 40 + np.random.default_rng(seed).normal(size=layer.Wg.shape)
    return Net(5, 3, layer, d, seed=seed)


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    return rng.normal(size=(40, 5)), rng.normal(size=(40, 3))


@pytest.mark.parametrize("kw", [
    dict(n_shared=1, k=1, cf=None),
    dict(n_shared=2, k=2, cf=0.8),                 # shared experts + real token dropping
    dict(n_shared=1, k=2, cf=1.0, aux_coef=0.05, z_coef=0.01),
])
def test_shared_expert_gradients_match_finite_differences(kw, data):
    assert _fd_check(_net(**kw), *data) < 1.0


def test_gradients_with_selection_bias(data):
    """The bias changes *which* experts are picked but gates stay the unbiased softmax."""
    net = _net(bias_rate=0.1, k=2, cf=1.0, n_shared=1)
    net.layer.bias = np.array([0.7, -0.4, 0.2, -0.9])
    assert _fd_check(net, *data) < 1.0


def test_shared_params_exposed_and_counted():
    a, b = _net(n_shared=0).layer, _net(n_shared=2).layer
    assert "sW1" not in a.params() and {"sW1", "sb1", "sW2", "sb2"} <= set(b.params())
    per_expert = 6 * 8 * 2 + 8 + 6
    assert b.n_params() - a.n_params() == 2 * per_expert
    assert b.active_params_per_token() - a.active_params_per_token() == 2 * per_expert


def test_zero_shared_is_bit_identical_to_v01_init(data):
    x, _ = data
    a, b = _net(n_shared=0).layer, _net(n_shared=0, bias_rate=0.0).layer
    h = np.random.default_rng(0).normal(size=(10, 6))
    assert np.array_equal(a.forward(h)[0], b.forward(h)[0])


def test_shared_experts_serve_tokens_that_routing_dropped():
    """With a tiny capacity most tokens are dropped by the router, but the shared path
    still transforms them, so no token passes through the layer unchanged."""
    h = np.random.default_rng(0).normal(size=(64, 6))
    routed = _net(k=1, cf=0.25).layer
    shared = _net(k=1, cf=0.25, n_shared=1).layer
    out_r, _ = routed.forward(h)
    out_s, _ = shared.forward(h)
    assert routed.last_stats["drop_rate"] > 0.5
    unchanged_r = np.all(out_r == h, axis=1).mean()
    unchanged_s = np.all(out_s == h, axis=1).mean()
    assert unchanged_r > 0.5 and unchanged_s == 0.0


def test_shared_output_adds_exactly():
    """out = routed-only out + shared-expert output (gate fixed at 1)."""
    h = np.random.default_rng(0).normal(size=(12, 6))
    base = _net(k=2, cf=None, n_shared=0).layer
    with_sh = _net(k=2, cf=None, n_shared=1).layer
    # same seed => identical router and routed experts (shared drawn after them)
    assert np.array_equal(base.ex.W1, with_sh.ex.W1) and np.array_equal(base.Wg, with_sh.Wg)
    y_sh, _ = with_sh.sh.forward(0, h)
    assert np.allclose(with_sh.forward(h)[0], base.forward(h)[0] + y_sh)


def test_bias_does_not_change_gates_only_selection():
    h = np.random.default_rng(3).normal(size=(30, 6))
    L = _net(bias_rate=0.05, k=1, cf=None, decisive=False).layer
    out0, c0 = L.forward(h)
    L.bias = np.array([0.0, 0.0, 0.0, 1e-9])      # too small to flip any selection
    out1, c1 = L.forward(h)
    assert np.allclose(out0, out1)
    L.bias = np.array([50.0, 0.0, 0.0, 0.0])      # now everyone goes to expert 0
    _, c2 = L.forward(h)
    assert np.all(c2["asg"]["selected"][:, 0])
    # the gate value is still the plain softmax probability of that expert
    assert np.allclose(c2["g"][:, 0], c2["p"][:, 0])


def test_update_bias_moves_against_load_error():
    L = _net(bias_rate=0.1, k=1, cf=None).layer
    L.last_stats = {"load": np.array([0.7, 0.1, 0.1, 0.1])}
    L.update_bias()
    assert L.bias[0] < 0 and np.all(L.bias[1:] > 0)
    assert np.allclose(L.bias, [-0.1, 0.1, 0.1, 0.1])
    L.bias_rate = 0.0
    before = L.bias.copy()
    L.update_bias()
    assert np.array_equal(L.bias, before)


def test_bias_balancing_rebalances_a_skewed_router():
    """Start from a router that sends ~everything to one expert; selection-bias updates alone
    (no loss gradient, router weights untouched) bring the load CV down."""
    rng = np.random.default_rng(0)
    L = MoELayer(6, 8, n_experts=4, k=1, capacity_factor=None, bias_rate=0.05, seed=0)
    L.Wg = rng.normal(size=L.Wg.shape) * 0.3
    L.Wg[:, 0] += 1.0                           # expert 0 is favoured for most inputs
    h = rng.normal(size=(512, 6)) + 0.5
    Wg0 = L.Wg.copy()
    cv = []
    for _ in range(150):
        L.forward(h)
        cv.append(load_stats(L.last_stats["load"])["cv"])
        L.update_bias()
    assert cv[0] > 0.8
    assert np.mean(cv[-20:]) < 0.5 * cv[0]
    assert np.array_equal(L.Wg, Wg0)


def test_bias_rate_requires_topk_and_validates():
    with pytest.raises(ValueError):
        MoELayer(6, 8, routing="expert_choice", bias_rate=0.1)
    with pytest.raises(ValueError):
        MoELayer(6, 8, n_shared=-1)


def test_fit_trains_with_shared_and_bias():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(128, 5))
    y = np.tanh(x @ rng.normal(size=(5, 3)))
    net = _net(n_shared=1, bias_rate=0.01, k=2, decisive=False)
    res = fit(net, x, y, x, y, steps=60, lr=5e-3)
    assert res.test_mse < 0.8 * float(np.mean(y ** 2))
    assert np.any(net.layer.bias != 0.0)
