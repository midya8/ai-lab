import numpy as np
import pytest

from gnnlab import (APPNP, GNN, JKNet, SGC, build_ops, fit, label_propagation, make_split,
                    propagate_ppr, sbm_graph, softmax_xent)
from gnnlab.decoupled import propagate_ppr_backward
from test_gnnlab import check_model, small_graph


def models(g):
    f, k = g.x.shape[1], g.k
    return {
        "appnp": APPNP(f, 6, k, k=4, alpha=0.15, seed=1),
        "appnp-deep-mlp": APPNP(f, 6, k, k=3, alpha=0.3, depth=3, seed=2),
        "sgc": SGC(f, k, k=3, seed=3),
        "jknet": JKNet(f, 5, k, depth=3, seed=4),
    }


@pytest.mark.parametrize("name", ["appnp", "appnp-deep-mlp", "sgc", "jknet"])
def test_gradients_match_finite_differences(name):
    g, ops = small_graph()
    check_model(models(g)[name], g, ops, np.arange(0, g.n, 2))


def test_ppr_backward_is_the_adjoint_of_forward():
    g, ops = small_graph(seed=2)
    rng = np.random.default_rng(0)
    z, d = rng.normal(size=(g.n, 4)), rng.normal(size=(g.n, 4))
    # <d, P z> == <P^T d, z> for the linear map P = propagate_ppr
    lhs = (d * propagate_ppr(z, ops.ahat, 5, 0.2)).sum()
    rhs = (propagate_ppr_backward(d, ops.ahat, 5, 0.2) * z).sum()
    assert np.isclose(lhs, rhs)


def test_appnp_with_k_zero_or_alpha_one_is_an_mlp():
    g, ops = small_graph()
    mlp = GNN("mlp", g.x.shape[1], 6, g.k, depth=2, seed=7)
    for kw in ({"k": 0, "alpha": 0.1}, {"k": 6, "alpha": 1.0}):
        a = APPNP(g.x.shape[1], 6, g.k, seed=7, **kw)
        assert np.allclose(a.forward(g.x, ops), mlp.forward(g.x, ops))


def test_ppr_converges_to_the_closed_form():
    g, ops = small_graph(seed=3)
    z = np.random.default_rng(1).normal(size=(g.n, 3))
    a = 0.2
    closed = a * np.linalg.solve(np.eye(g.n) - (1 - a) * ops.ahat, z)
    assert np.allclose(propagate_ppr(z, ops.ahat, 200, a), closed, atol=1e-8)


def test_ppr_keeps_a_fixed_share_of_the_input_at_any_depth():
    """alpha > 0 means the output never collapses: it stays correlated with Z as K grows."""
    g, ops = small_graph(seed=4)
    z = np.random.default_rng(2).normal(size=(g.n, 4))
    far = propagate_ppr(z, ops.ahat, 500, 0.1)
    plain = np.linalg.matrix_power(ops.ahat, 500) @ z
    cos = lambda a, b: float((a * b).sum() / np.linalg.norm(a) / np.linalg.norm(b))
    assert cos(far, z) > cos(plain, z) + 0.1


def test_sgc_matches_manual_smoothing_and_k0_is_logistic_regression():
    g, ops = small_graph()
    m = SGC(g.x.shape[1], g.k, k=2, seed=0)
    manual = ops.ahat @ (ops.ahat @ g.x)
    out = m.forward(g.x, ops)
    assert np.allclose(out, manual @ m.layers[0].params["W"] + m.layers[0].params["b"])
    m0 = SGC(g.x.shape[1], g.k, k=0, seed=0)
    assert np.allclose(m0.smoothed(g.x, ops), g.x)


def test_sgc_cache_is_keyed_on_the_inputs():
    g, ops = small_graph()
    m = SGC(g.x.shape[1], g.k, k=2)
    a = m.smoothed(g.x, ops)
    assert m.smoothed(g.x, ops) is a                       # cached
    x2 = g.x * 2.0
    assert np.allclose(m.smoothed(x2, ops), 2.0 * a)       # new input -> recomputed


@pytest.mark.parametrize("name", ["appnp", "sgc", "jknet"])
def test_permutation_equivariance(name):
    g, ops = small_graph(seed=5)
    perm = np.random.default_rng(0).permutation(g.n)
    ops_p = build_ops(g.adj[np.ix_(perm, perm)])
    m = models(g)[name]
    out = m.forward(g.x, ops)
    out_p = m.forward(g.x[perm], ops_p)
    assert np.allclose(out[perm], out_p)


def test_jknet_concatenates_every_layer():
    g, ops = small_graph()
    m = JKNet(g.x.shape[1], 5, g.k, depth=3)
    m.forward(g.x, ops)
    assert len(m.hidden_states) == 3 and m.layers[-1].params["W"].shape == (15, g.k)


def test_state_roundtrip_and_validation():
    g, ops = small_graph()
    m = APPNP(g.x.shape[1], 6, g.k, seed=0)
    s = m.state()
    out = m.forward(g.x, ops).copy()
    for _, p, _ in m.named_params():
        p += 1.0
    m.load_state(s)
    assert np.allclose(m.forward(g.x, ops), out)
    assert m.n_params() == sum(p.size for p in s.values())
    with pytest.raises(ValueError):
        APPNP(3, 4, 2, k=-1)
    with pytest.raises(ValueError):
        APPNP(3, 4, 2, alpha=1.5)
    with pytest.raises(ValueError):
        SGC(3, 2, k=-1)
    with pytest.raises(ValueError):
        JKNet(3, 4, 2, depth=0)


def test_label_propagation_closed_form_and_limits():
    g, ops = small_graph(seed=6)
    train = np.arange(0, g.n, 3)
    a = 0.8
    y0 = np.zeros((g.n, g.k))
    y0[train, g.y[train]] = 1.0
    closed = (1 - a) * np.linalg.solve(np.eye(g.n) - a * ops.ahat, y0)
    assert np.allclose(label_propagation(ops, g.y, train, g.k, alpha=a, iters=400), closed, atol=1e-8)
    f0 = label_propagation(ops, g.y, train, g.k, alpha=0.0)
    assert np.allclose(f0, y0)                              # no spreading: just the seeds
    with pytest.raises(ValueError):
        label_propagation(ops, g.y, train, g.k, alpha=1.0)


def test_label_propagation_isolated_nodes_stay_blank():
    adj = np.zeros((6, 6))
    adj[0, 1] = adj[1, 0] = adj[1, 2] = adj[2, 1] = 1.0   # nodes 3-5 isolated
    y = np.array([0, 0, 0, 1, 1, 1])
    f = label_propagation(build_ops(adj), y, np.array([0]), 2)
    assert f[1].sum() > 0 and f[3:, :].max() == 0.0


def test_label_propagation_ignores_features_and_wins_when_homophilous():
    g = sbm_graph(n=300, k=3, avg_degree=10, homophily=0.9, feat_dim=4, feat_sep=0.0, seed=1)
    ops = build_ops(g.adj)
    tr, va, te = make_split(g.y, per_class=15, n_val=60)
    f = label_propagation(ops, g.y, tr, g.k)
    assert (f[te].argmax(1) == g.y[te]).mean() > 0.9        # features are pure noise here


def test_all_decoupled_models_train_with_fit():
    g = sbm_graph(n=200, k=3, avg_degree=8, homophily=0.85, feat_dim=8, seed=0)
    ops = build_ops(g.adj)
    tr, va, te = make_split(g.y, per_class=15, n_val=50)
    mlp_acc = fit(GNN("mlp", 8, 16, 3), g.x, g.y, ops, tr, va, te, epochs=100).test_acc
    for m in (APPNP(8, 16, 3, k=8, seed=0), SGC(8, 3, k=2, seed=0), JKNet(8, 16, 3, depth=3, seed=0)):
        r = fit(m, g.x, g.y, ops, tr, va, te, epochs=100)
        assert r.losses[-1] < r.losses[0]
        assert r.test_acc > mlp_acc + 0.05, type(m).__name__
