import numpy as np
import pytest
from ntklab import MLP, gd_predict, kernel_mean_prediction, stable_lr, train_gd, kernel_change, residual_modes, ntk


def data(n=10, seed=0):
    rng = np.random.default_rng(seed)
    phi = np.sort(rng.uniform(0, 2 * np.pi, n))
    X = np.stack([np.cos(phi), np.sin(phi)], 1)
    return X, np.sin(phi) + 0.5 * np.sin(3 * phi), phi


def test_gd_predict_matches_explicit_linearised_gd():
    X, y, _ = data()
    Xt = np.stack([np.cos(np.linspace(0, 6, 7)), np.sin(np.linspace(0, 6, 7))], 1)
    net = MLP(2, 20, 2, seed=1)
    J, Jt = net.jacobian(X), net.jacobian(Xt)
    K, Kt = J @ J.T, Jt @ J.T
    lr = stable_lr(K, 0.8)
    steps = 37
    # explicit parameter-space GD on the linearised model
    delta = np.zeros(J.shape[1]); f0, f0t = net(X), net(Xt)
    n = len(y)
    for _ in range(steps):
        r = f0 + J @ delta - y
        delta -= lr * J.T @ r / n
    tr, te = gd_predict(K, Kt, f0, f0t, y, lr, steps)
    assert np.allclose(tr, f0 + J @ delta, atol=1e-9)
    assert np.allclose(te, f0t + Jt @ delta, atol=1e-9)


def test_gd_predict_zero_steps_returns_init():
    X, y, _ = data()
    net = MLP(2, 10, 1, seed=2)
    K = net.ntk(X)
    tr, te = gd_predict(K, K, net(X), net(X), y, 1.0, 0)
    assert np.allclose(tr, net(X)) and np.allclose(te, net(X))


def test_gd_converges_to_interpolation_and_kernel_mean():
    X, y, _ = data()
    K = ntk(X, depth=2)
    lr = stable_lr(K, 0.9)
    z = np.zeros(len(y))
    tr, te = gd_predict(K, K, z, z, y, lr, 200000)
    assert np.allclose(tr, y, atol=1e-6)
    assert np.allclose(te, kernel_mean_prediction(K, K, y), atol=1e-6)


def test_wide_net_training_tracks_linearised_prediction():
    X, y, phi = data()
    Xt = np.stack([np.cos(np.linspace(0, 6, 15)), np.sin(np.linspace(0, 6, 15))], 1)
    gaps = {}
    for w in (16, 2048):
        net = MLP(2, w, 2, seed=7)
        K, Kt = net.ntk(X), net.ntk(Xt, X)
        lr = stable_lr(K, 0.5)
        res = train_gd(net, X, y, lr, 400)
        _, lin = gd_predict(K, Kt, net(X), net(Xt), y, lr, 400)
        gaps[w] = np.linalg.norm(res.predict(Xt) - lin) / np.linalg.norm(lin)
    assert gaps[2048] < 0.05
    assert gaps[2048] < 0.5 * gaps[16]


def test_training_reduces_loss_and_is_deterministic():
    X, y, _ = data()
    net = MLP(2, 64, 2, seed=3)
    lr = stable_lr(net.ntk(X), 0.5)
    a = train_gd(net, X, y, lr, 300, record_every=50)
    b = train_gd(net, X, y, lr, 300, record_every=50)
    assert a.losses == b.losses
    assert a.losses[-1] < 0.05 * a.losses[0]
    assert np.allclose(net.get_flat(), a.net0.get_flat())      # input net untouched


def test_alpha_scaling_makes_training_lazier():
    X, y, _ = data()
    net = MLP(2, 64, 2, seed=5)
    K0 = net.ntk(X)
    lr = stable_lr(K0, 0.5)
    moves = {}
    for alpha in (1.0, 300.0):
        res = train_gd(net, X, y, lr, 600, alpha=alpha, center=True)
        moves[alpha] = kernel_change(res.net.ntk(X), K0)
    assert moves[300.0] < 0.05 * moves[1.0]


def test_alpha_training_function_dynamics_match_for_large_alpha():
    X, y, _ = data()
    net = MLP(2, 256, 2, seed=6)
    K0 = net.ntk(X)
    lr = stable_lr(K0, 0.5)
    z = np.zeros(len(y))
    r = train_gd(net, X, y, lr, 150, alpha=50.0, center=True)
    tr, _ = gd_predict(K0, K0, z, z, y, lr, 150)
    assert np.linalg.norm(r.predict(X) - tr) / np.linalg.norm(tr) < 0.02


def test_residual_modes_reconstruct():
    X, y, _ = data()
    K = ntk(X)
    lam, U, c = residual_modes(K, y)
    assert np.all(np.diff(lam) <= 1e-12)
    assert np.allclose(U @ c, y)


def test_stable_lr_is_actually_stable_and_larger_is_not():
    X, y, _ = data()
    K = ntk(X)
    z = np.zeros(len(y))
    ok, _ = gd_predict(K, K, z, z, y, stable_lr(K, 0.95), 2000)
    bad, _ = gd_predict(K, K, z, z, y, stable_lr(K, 1.2), 2000)
    assert np.linalg.norm(ok - y) < np.linalg.norm(y)
    assert np.linalg.norm(bad - y) > 1e3
