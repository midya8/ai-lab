import numpy as np
import pytest
from ntklab import MLP


def fd_grad(f, theta, eps=1e-6):
    g = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta); e[i] = eps
        g[i] = (f(theta + e) - f(theta - e)) / (2 * eps)
    return g


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_jacobian_matches_finite_differences(depth):
    rng = np.random.default_rng(depth)
    net = MLP(3, 5, depth, sb2=0.3, seed=depth)
    X = rng.standard_normal((4, 3))
    J = net.jacobian(X)
    theta = net.get_flat()
    assert J.shape == (4, net.n_params())
    for i in range(4):
        def f(th, i=i):
            m = net.copy(); m.set_flat(th)
            return m(X[i:i + 1])[0]
        num = fd_grad(f, theta)
        assert np.allclose(J[i], num, rtol=1e-5, atol=1e-7)


def test_loss_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    net = MLP(2, 6, 2, seed=3)
    X = rng.standard_normal((5, 2)); y = rng.standard_normal(5)
    f, cache = net.forward(X)
    g = net.grads_flat(cache, (f - y) / 5)

    def loss(th):
        m = net.copy(); m.set_flat(th)
        return 0.5 * np.mean((m(X) - y) ** 2)
    assert np.allclose(g, fd_grad(loss, net.get_flat()), rtol=1e-5, atol=1e-8)


def test_wrong_backward_is_detected():
    """The finite-difference check must fail for a deliberately broken backward pass."""
    rng = np.random.default_rng(4)
    net = MLP(2, 6, 2, seed=4)
    X = rng.standard_normal((5, 2))
    J = net.jacobian(X)
    good = J.copy()
    bad = J.copy(); bad[:, :4] *= 1.05
    theta = net.get_flat()

    def f(th):
        m = net.copy(); m.set_flat(th); return m(X[:1])[0]
    num = fd_grad(f, theta)
    assert np.allclose(good[0], num, rtol=1e-5, atol=1e-7)
    assert not np.allclose(bad[0], num, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_empirical_ntk_equals_jacobian_gram(depth):
    rng = np.random.default_rng(5)
    net = MLP(3, 7, depth, sb2=0.2, seed=5)
    A = rng.standard_normal((6, 3)); B = rng.standard_normal((4, 3))
    JA, JB = net.jacobian(A), net.jacobian(B)
    assert np.allclose(net.ntk(A), JA @ JA.T, rtol=1e-9, atol=1e-10)
    assert np.allclose(net.ntk(A, B), JA @ JB.T, rtol=1e-9, atol=1e-10)


def test_flat_roundtrip_and_copy_independence():
    net = MLP(2, 4, 2, seed=1)
    th = net.get_flat()
    c = net.copy(); c.W[0][0, 0] += 1.0
    assert np.allclose(net.get_flat(), th)
    c.set_flat(th)
    assert np.allclose(c.get_flat(), th)
    with pytest.raises(ValueError):
        c.set_flat(th[:-1])


def test_forward_scale_preserves_preactivation_variance():
    # with He-style sw2=2 hidden scaling, h-variance stays ~ |x|^2 * 2/d + sb2 through depth
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 50)); x /= np.linalg.norm(x) / np.sqrt(50)
    vars_ = []
    for s in range(200):
        net = MLP(50, 400, 4, sb2=0.0, seed=s)
        _, c = net.forward(x)
        vars_.append(np.mean(c.pre[-2] ** 2))
    assert 0.8 < np.mean(vars_) / 2.0 < 1.25
