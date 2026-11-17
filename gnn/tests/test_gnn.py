import numpy as np
import pytest

from gnn import (GNN, build_ops, dirichlet_energy, edge_homophily, fit, make_split,
                    mean_pairwise_cosine, sbm_graph, softmax_xent)
from gnn.model import pairnorm, pairnorm_backward


def small_graph(seed=0, n=24):
    g = sbm_graph(n=n, k=3, avg_degree=4, homophily=0.7, feat_dim=5, seed=seed)
    return g, build_ops(g.adj)


def numerical_grad(f, p, eps=1e-6):
    g = np.zeros_like(p)
    it = np.nditer(p, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = p[i]
        p[i] = old + eps
        fp = f()
        p[i] = old - eps
        fm = f()
        p[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def check_model(model, g, ops, idx, tol=1e-4):
    def loss():
        return softmax_xent(model.forward(g.x, ops), g.y, idx)[0]

    logits = model.forward(g.x, ops)
    _, dl = softmax_xent(logits, g.y, idx)
    model.backward(dl)
    for name, p, grad in model.named_params():
        num = numerical_grad(loss, p)
        # elementwise rtol plus a small atol: biases feeding PairNorm have a true gradient of 0
        bad = np.abs(num - grad) > 1e-7 + tol * np.abs(num)
        assert not bad.any(), (name, np.abs(num - grad).max(), np.abs(num).max())


# ---- graph construction --------------------------------------------------------------
def test_sbm_is_symmetric_loopless_and_deterministic():
    g1 = sbm_graph(seed=3)
    g2 = sbm_graph(seed=3)
    assert np.array_equal(g1.adj, g2.adj) and np.array_equal(g1.x, g2.x)
    assert np.array_equal(g1.adj, g1.adj.T)
    assert np.all(np.diag(g1.adj) == 0)
    assert not np.array_equal(g1.adj, sbm_graph(seed=4).adj)


@pytest.mark.parametrize("h", [0.2, 0.5, 0.9])
def test_sbm_hits_target_homophily_and_degree(h):
    g = sbm_graph(n=800, homophily=h, avg_degree=10, seed=1)
    assert abs(edge_homophily(g.adj, g.y) - h) < 0.04
    assert abs(g.adj.sum(1).mean() - 10) < 1.0


def test_sbm_validates_arguments():
    with pytest.raises(ValueError):
        sbm_graph(homophily=1.5)
    with pytest.raises(ValueError):
        sbm_graph(n=4, k=4)


def test_split_is_disjoint_stratified_and_exhaustive():
    g = sbm_graph(n=300, k=3, seed=0)
    tr, va, te = make_split(g.y, per_class=10, n_val=50, seed=2)
    assert len(tr) == 30 and len(va) == 50
    assert len(set(tr) | set(va) | set(te)) == 300
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert all((g.y[tr] == c).sum() == 10 for c in range(3))
    with pytest.raises(ValueError):
        make_split(g.y, per_class=10, n_val=10_000)


def test_ops_normalisation():
    g, ops = small_graph()
    assert np.allclose(ops.ahat, ops.ahat.T)
    # D~^-1/2 (A+I) D~^-1/2 has spectral radius exactly 1 with eigenvector sqrt(d~)
    d = (g.adj + np.eye(g.n)).sum(1)
    assert np.allclose(ops.ahat @ np.sqrt(d), np.sqrt(d))
    assert np.linalg.eigvalsh(ops.ahat).max() == pytest.approx(1.0)
    deg = g.adj.sum(1)
    assert np.allclose(ops.mean.sum(1)[deg > 0], 1.0)
    assert np.all(np.diag(ops.mask))


def test_ops_rejects_asymmetric_and_handles_isolated_nodes():
    a = np.zeros((4, 4))
    a[0, 1] = 1
    with pytest.raises(ValueError):
        build_ops(a)
    a[1, 0] = 1  # nodes 2, 3 isolated
    ops = build_ops(a)
    assert np.all(ops.mean[2] == 0) and np.isfinite(ops.ahat).all()
    assert ops.ahat[2, 2] == 1.0


# ---- gradient checks ------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["mlp", "gcn", "sage", "gat"])
@pytest.mark.parametrize("depth", [1, 3])
def test_gradients_match_finite_differences(kind, depth):
    g, ops = small_graph(seed=1)
    model = GNN(kind, 5, 8, 3, depth=depth, seed=2)
    check_model(model, g, ops, np.arange(0, g.n, 2))


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat"])
def test_gradients_with_residual_and_pairnorm(kind):
    g, ops = small_graph(seed=2)
    model = GNN(kind, 5, 8, 3, depth=4, seed=3, residual=True, pairnorm_scale=1.5)
    check_model(model, g, ops, np.arange(g.n))


def test_gat_gradient_with_multiple_heads_and_isolated_nodes():
    g, ops = small_graph(seed=5, n=20)
    g.adj[:3, :] = 0
    g.adj[:, :3] = 0
    ops = build_ops(g.adj)
    model = GNN("gat", 5, 8, 3, depth=2, seed=4, heads=2)
    check_model(model, g, ops, np.arange(g.n))


def test_pairnorm_properties_and_backward():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(10, 4)) + 5.0
    y, cache = pairnorm(x, 2.0)
    assert np.allclose(y.mean(0), 0.0)
    assert (y * y).sum(1).mean() == pytest.approx(4.0, rel=1e-6)
    w = rng.normal(size=y.shape)

    def f():
        return float((pairnorm(x, 2.0)[0] * w).sum())

    num = numerical_grad(f, x)
    assert np.allclose(num, pairnorm_backward(w, cache, 2.0), atol=1e-6)


# ---- layer semantics ------------------------------------------------------------------
def test_gat_attention_rows_are_distributions_supported_on_neighbours():
    g, ops = small_graph(seed=3)
    model = GNN("gat", 5, 8, 3, depth=2, seed=1)
    model.forward(g.x, ops)
    for layer in model.layers:
        a = layer.attn
        assert np.allclose(a.sum(2), 1.0)
        assert np.all(a[:, ~ops.mask] == 0)


@pytest.mark.parametrize("kind", ["gcn", "sage", "gat"])
def test_permutation_equivariance(kind):
    g, ops = small_graph(seed=4)
    model = GNN(kind, 5, 8, 3, depth=2, seed=7)
    out = model.forward(g.x, ops)
    perm = np.random.default_rng(0).permutation(g.n)
    ops_p = build_ops(g.adj[np.ix_(perm, perm)])
    out_p = model.forward(g.x[perm], ops_p)
    assert np.allclose(out[perm], out_p, atol=1e-10)


def test_mlp_ignores_the_graph():
    g, ops = small_graph(seed=6)
    model = GNN("mlp", 5, 8, 3, depth=2, seed=1)
    a = model.forward(g.x, ops)
    b = model.forward(g.x, build_ops(np.zeros_like(g.adj)))
    assert np.array_equal(a, b)


def test_sage_separates_self_from_neighbours():
    # a node's own features pass through Ws only: zeroing Wn makes SAGE a per-node map.
    g, ops = small_graph(seed=8)
    model = GNN("sage", 5, 8, 3, depth=1, seed=1)
    model.layers[0].params["Wn"][...] = 0.0
    out = model.forward(g.x, ops)
    expect = g.x @ model.layers[0].params["Ws"] + model.layers[0].params["b"]
    assert np.allclose(out, expect)


def test_model_validation():
    with pytest.raises(ValueError):
        GNN("transformer", 4, 8, 2)
    with pytest.raises(ValueError):
        GNN("gat", 4, 10, 2, heads=4)
    with pytest.raises(ValueError):
        GNN("gcn", 4, 8, 2, depth=0)


def test_state_roundtrip_and_param_count():
    model = GNN("gcn", 5, 8, 3, depth=2)
    assert model.n_params() == 5 * 8 + 8 + 8 * 3 + 3
    s = model.state()
    for _, p, _ in model.named_params():
        p += 1.0
    model.load_state(s)
    for n, p, _ in model.named_params():
        assert np.array_equal(p, s[n])


# ---- metrics --------------------------------------------------------------------------
def test_dirichlet_energy_zero_on_sqrt_degree_and_scale_invariant():
    g, ops = small_graph(seed=9)
    d = (g.adj + np.eye(g.n)).sum(1)
    h = np.outer(np.sqrt(d), np.array([1.0, -2.0, 0.5]))
    assert dirichlet_energy(h, ops) == pytest.approx(0.0, abs=1e-12)
    x = np.random.default_rng(0).normal(size=(g.n, 4))
    assert dirichlet_energy(3.0 * x, ops) == pytest.approx(dirichlet_energy(x, ops))
    assert 0.0 <= dirichlet_energy(x, ops) <= 2.0 + 1e-9
    assert dirichlet_energy(np.zeros((g.n, 2)), ops) == 0.0


def test_propagation_drives_energy_to_zero_on_a_connected_graph():
    # ring + chord keeps it connected; repeated Ahat multiplication oversmooths.
    n = 30
    a = np.zeros((n, n))
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1
    a[0, 15] = a[15, 0] = 1
    ops = build_ops(a)
    h = np.random.default_rng(0).normal(size=(n, 3))
    energies = []
    for _ in range(200):
        h = ops.ahat @ h
        energies.append(dirichlet_energy(h, ops))
    assert energies[-1] < 1e-3 < energies[0]
    assert all(b <= a + 1e-12 for a, b in zip(energies, energies[1:]))


def test_mean_pairwise_cosine_extremes():
    assert mean_pairwise_cosine(np.ones((5, 3))) == pytest.approx(1.0)
    h = np.array([[1.0, 0], [-1.0, 0]])
    assert mean_pairwise_cosine(h) == pytest.approx(-1.0)


# ---- training -------------------------------------------------------------------------
def test_training_reduces_loss_and_restores_best_epoch():
    g = sbm_graph(n=200, k=3, homophily=0.85, feat_dim=8, seed=0)
    ops = build_ops(g.adj)
    tr, va, te = make_split(g.y, per_class=10, n_val=60, seed=0)
    model = GNN("gcn", 8, 16, 3, depth=2, seed=0)
    res = fit(model, g.x, g.y, ops, tr, va, te, epochs=80)
    assert res.losses[-1] < res.losses[0]
    assert 1 <= res.best_epoch <= 80
    assert res.val_acc >= 0.8
    # the restored model reproduces the reported test accuracy exactly
    logits = model.forward(g.x, ops)
    assert (logits[te].argmax(1) == g.y[te]).mean() == pytest.approx(res.test_acc)


def test_patience_stops_early():
    g = sbm_graph(n=200, k=3, homophily=0.85, feat_dim=8, seed=0)
    ops = build_ops(g.adj)
    tr, va, te = make_split(g.y, per_class=10, n_val=60, seed=0)
    res = fit(GNN("gcn", 8, 16, 3, seed=0), g.x, g.y, ops, tr, va, te, epochs=500, patience=10)
    assert len(res.losses) < 500


def test_gcn_beats_mlp_when_homophilous_and_loses_when_heterophilous():
    def run(kind, h):
        accs = []
        for s in range(3):
            g = sbm_graph(n=300, k=4, avg_degree=10, homophily=h, feat_dim=16, seed=s)
            ops = build_ops(g.adj)
            tr, va, te = make_split(g.y, per_class=15, n_val=80, seed=s)
            accs.append(fit(GNN(kind, 16, 32, 4, seed=s), g.x, g.y, ops, tr, va, te).test_acc)
        return float(np.mean(accs))

    assert run("gcn", 0.9) > run("mlp", 0.9) + 0.15
    assert run("gcn", 0.1) < run("mlp", 0.1) - 0.1


def test_softmax_xent_gradient_only_touches_selected_rows():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(6, 3))
    y = np.array([0, 1, 2, 0, 1, 2])
    idx = np.array([1, 4])
    loss, grad = softmax_xent(logits, y, idx)
    assert np.all(grad[[0, 2, 3, 5]] == 0)
    num = numerical_grad(lambda: softmax_xent(logits, y, idx)[0], logits)
    assert np.allclose(num, grad, atol=1e-7)
