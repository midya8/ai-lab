import numpy as np
import pytest

from moelab import DenseFFN, MoELayer, Net, fit, load_stats, make_task, routing_nmi, routing_purity, sample_task


def _net(routing="topk", k=2, cf=1.25, aux=0.0, z=0.0, E=4, renorm=None, seed=0, d=6):
    layer = MoELayer(d, 8, n_experts=E, k=k, capacity_factor=cf, routing=routing,
                     aux_coef=aux, z_coef=z, renorm=renorm, seed=seed)
    layer.Wg = layer.Wg * 40 + np.random.default_rng(seed).normal(size=layer.Wg.shape)  # decisive router
    return Net(5, 3, layer, d, seed=seed)


def _fd_check(net, x, y, n_probe=6, eps=1e-6, seed=0):
    _, _, grads = net.loss_and_grads(x, y)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for name, p in net.params().items():
        flat = p.reshape(-1)
        for i in rng.choice(flat.size, size=min(n_probe, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = net.loss_and_grads(x, y)[0]
            flat[i] = old - eps
            lm = net.loss_and_grads(x, y)[0]
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            an = grads[name].reshape(-1)[i]
            worst = max(worst, abs(fd - an) / (1e-5 + 1e-4 * abs(fd)))
    return worst


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    return rng.normal(size=(40, 5)), rng.normal(size=(40, 3))


@pytest.mark.parametrize("kw", [
    dict(k=1, cf=None),
    dict(k=2, cf=None),
    dict(k=2, cf=0.8),                      # real token dropping
    dict(k=1, cf=0.7),
    dict(k=2, cf=1.25, renorm=False),
    dict(k=2, cf=1.0, aux=0.05, z=0.01),    # auxiliary losses differentiate correctly
    dict(routing="expert_choice", k=2, cf=1.0),
    dict(routing="expert_choice", k=1, cf=0.6, z=0.01),
])
def test_gradients_match_finite_differences(kw, data):
    net = _net(**kw)
    assert _fd_check(net, *data) < 1.0


def test_dense_ffn_gradients(data):
    net = Net(5, 3, DenseFFN(6, 16), 6)
    assert _fd_check(net, *data) < 1.0


def test_capacity_drops_tokens_and_respects_limit(data):
    x, y = data
    net = _net(k=1, cf=0.5)
    net.forward(x)
    st = net.layer.last_stats
    C = net.layer.capacity(len(x))
    assert C == int(np.ceil(0.5 * 40 / 4))
    assert st["drop_rate"] > 0
    # hard check on kept counts
    out, cache = net.layer.forward(net.forward(x)[1][1])
    assert cache["asg"]["kept"].sum(axis=0).max() <= C


def test_no_capacity_means_no_drops(data):
    net = _net(k=2, cf=None)
    net.forward(data[0])
    assert net.layer.last_stats["drop_rate"] == 0.0


def test_topk_selects_k_experts_and_gates_renormalise():
    layer = MoELayer(6, 8, n_experts=5, k=3, capacity_factor=None, seed=0)
    h = np.random.default_rng(0).normal(size=(20, 6))
    _, cache = layer.forward(h)
    assert (cache["asg"]["selected"].sum(1) == 3).all()
    assert np.allclose(cache["g"].sum(1), 1.0)


def test_k1_default_keeps_raw_probability_gate():
    layer = MoELayer(6, 8, n_experts=4, k=1, capacity_factor=None, seed=0)
    assert layer.renorm is False
    h = np.random.default_rng(0).normal(size=(10, 6))
    _, cache = layer.forward(h)
    assert np.allclose(cache["g"].sum(1), cache["p"].max(1))


def test_expert_choice_is_perfectly_balanced(data):
    net = _net(routing="expert_choice", k=2, cf=1.0)
    net.forward(data[0])
    st = net.layer.last_stats
    assert np.allclose(st["load"], 0.25)
    assert st["drop_rate"] == 0.0
    assert load_stats(st["load"])["cv"] < 1e-9


def test_priority_first_choices_before_second_choices():
    # capacity 1 per expert: second choices must never evict a first choice
    layer = MoELayer(4, 4, n_experts=4, k=2, capacity_factor=1.0, seed=0)
    p = np.array([[0.7, 0.2, 0.05, 0.05], [0.6, 0.3, 0.05, 0.05],
                  [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7]])
    asg = layer._route_topk(p, C=2)
    assert asg["kept"][0, 0] and asg["kept"][1, 0]     # both first choices of expert 0 fit
    assert asg["kept"][2, 2] and asg["kept"][3, 3]
    assert asg["kept"].sum(axis=0).max() <= 2


def test_dropped_token_passes_through_residual():
    layer = MoELayer(4, 4, n_experts=2, k=1, capacity_factor=0.5, seed=0)
    layer.Wg = np.array([[50.0, -50.0]] * 4)             # everyone prefers expert 0
    h = np.abs(np.random.default_rng(0).normal(size=(8, 4)))
    out, cache = layer.forward(h)
    dropped = ~cache["asg"]["kept"].any(axis=1)
    assert dropped.sum() == 8 - layer.capacity(8)
    assert np.allclose(out[dropped], h[dropped])


def test_load_balance_loss_is_one_when_uniform_and_E_when_collapsed():
    layer = MoELayer(4, 4, n_experts=4, k=1, capacity_factor=None, aux_coef=1.0, seed=0)
    layer.Wg = np.zeros((4, 4))
    layer.forward(np.random.default_rng(0).normal(size=(64, 4)))
    # uniform probabilities, ties -> expert 0 gets all top-1: f=(1,0,0,0), P=1/4 -> lb = E*1*(1/4) = 1
    assert layer.last_stats["lb"] == pytest.approx(1.0)
    layer.Wg = np.zeros((4, 4))
    layer.Wg[:, 0] = 50.0
    layer.forward(np.abs(np.random.default_rng(0).normal(size=(64, 4))) + 0.5)
    assert layer.last_stats["lb"] == pytest.approx(4.0, rel=1e-3)   # total collapse


def test_aux_loss_gradient_pushes_towards_balance():
    rng = np.random.default_rng(0)
    layer = MoELayer(4, 4, n_experts=4, k=1, capacity_factor=None, aux_coef=1.0, seed=0)
    layer.Wg = rng.normal(size=(4, 4)) * 0.3
    layer.Wg[:, 0] += 1.0
    h = np.abs(rng.normal(size=(200, 4)))
    layer.forward(h)
    lb0, cv0 = layer.last_stats["lb"], load_stats(layer.last_stats["load"])["cv"]
    for _ in range(100):
        out, cache = layer.forward(h)
        _, g = layer.backward(np.zeros_like(out), cache)
        layer.Wg -= 0.5 * g["Wg"]
    layer.forward(h)
    assert layer.last_stats["lb"] < lb0
    assert load_stats(layer.last_stats["load"])["cv"] < 0.5 * cv0


def test_jitter_only_in_train_mode():
    layer = MoELayer(4, 4, n_experts=4, k=2, jitter=0.1, seed=0)
    h = np.random.default_rng(0).normal(size=(16, 4))
    a, _ = layer.forward(h)
    b, _ = layer.forward(h)
    assert not np.allclose(a, b)
    layer.train_mode = False
    c, _ = layer.forward(h)
    d, _ = layer.forward(h)
    assert np.allclose(c, d)


def test_metrics_nmi_and_purity():
    z = np.repeat(np.arange(4), 25)
    assert routing_nmi(z, z) == pytest.approx(1.0)
    assert routing_nmi((z + 2) % 4, z) == pytest.approx(1.0)       # relabelling is irrelevant
    assert routing_nmi(np.zeros_like(z), z) == 0.0
    assert routing_purity(z, z) == 1.0
    rng = np.random.default_rng(0)
    assert routing_nmi(rng.integers(0, 4, 4000), np.repeat(np.arange(4), 1000)) < 0.01
    assert routing_purity(np.arange(100) % 2, np.zeros(100, int)) == 0.5


def test_load_stats():
    s = load_stats(np.full(8, 1 / 8))
    assert s["cv"] == pytest.approx(0) and s["norm_entropy"] == pytest.approx(1) and s["dead_experts"] == 0
    s = load_stats(np.array([1.0, 0, 0, 0]))
    assert s["dead_experts"] == 3 and s["max_over_mean"] == pytest.approx(4.0) and s["norm_entropy"] == 0


def test_task_is_cluster_structured_and_deterministic():
    t = make_task(n_clusters=4, seed=3)
    x1, y1, z1 = sample_task(t, 100, seed=7)
    x2, y2, z2 = sample_task(t, 100, seed=7)
    assert np.array_equal(x1, x2) and np.array_equal(z1, z2)
    assert x1.shape == (100, 8) and y1.shape == (100, 4)
    assert np.abs(y1).max() < 1.2


def test_training_reduces_loss_and_learns_to_route():
    t = make_task(n_clusters=4, seed=0)
    x, y, z = sample_task(t, 512, 1)
    xt, yt, _ = sample_task(t, 512, 2)
    layer = MoELayer(16, 24, n_experts=4, k=1, capacity_factor=1.5, aux_coef=0.02, seed=0)
    net = Net(8, 4, layer, 16, seed=0)
    base = float(np.mean((net.predict(xt) - yt) ** 2))
    res = fit(net, x, y, xt, yt, steps=250, lr=5e-3)
    assert res.test_mse < 0.5 * base
    net.predict(x)
    assert routing_nmi(layer.last_stats["top1"], z) > 0.5


def test_params_counts():
    layer = MoELayer(6, 8, n_experts=4, k=2)
    assert layer.n_params() == 6 * 4 + 4 * (6 * 8 + 8 + 8 * 6 + 6)
    assert layer.active_params_per_token() == 6 * 4 + 2 * (6 * 8 * 2 + 8 + 6)
    assert DenseFFN(6, 16).n_params() == 6 * 16 + 16 + 16 * 6 + 6


def test_bad_args():
    with pytest.raises(ValueError):
        MoELayer(4, 4, routing="nope")
    with pytest.raises(ValueError):
        MoELayer(4, 4, n_experts=2, k=3)
