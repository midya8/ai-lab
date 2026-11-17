import numpy as np
import pytest

from normflow import (Coupling, Flow, density_grid_integral, fit, bridge_mass, nll_gap,
                     gaussian_ring, two_moons, checkerboard, spiral)
from normflow.layers import ActNorm, Permutation


def rand_flow(d=4, layers=3, w2=0.4, **kw):
    f = Flow(d, layers, 8, seed=1, zero_init=False, **kw)
    rng = np.random.default_rng(0)
    for l in f.layers:                       # make every layer non-trivial
        if isinstance(l, ActNorm):
            l.params["log_s"] = 0.3 * rng.standard_normal(d)
            l.params["b"] = 0.3 * rng.standard_normal(d)
            l.initialised = True
        if isinstance(l, Coupling):
            l.params["W2"] = w2 * rng.standard_normal(l.params["W2"].shape)
            l.params["b2"] = 0.1 * rng.standard_normal(l.params["b2"].shape)
    return f


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("d", [2, 5])
def test_inverse_roundtrip(additive, d):
    f = rand_flow(d, 4, additive=additive)
    x = np.random.default_rng(2).standard_normal((50, d))
    z, _, _ = f.forward(x)
    assert np.allclose(f.inverse(z), x, atol=1e-9)


def test_logdet_matches_numerical_jacobian():
    f = rand_flow(3, 3)
    rng = np.random.default_rng(3)
    x = rng.standard_normal((6, 3))
    _, logdet, _ = f.forward(x)
    eps = 1e-6
    for n in range(len(x)):
        J = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = eps
            zp, _, _ = f.forward((x[n] + e)[None])
            zm, _, _ = f.forward((x[n] - e)[None])
            J[:, j] = (zp - zm)[0] / (2 * eps)
        assert np.log(abs(np.linalg.det(J))) == pytest.approx(logdet[n], abs=1e-6)


def test_additive_coupling_is_volume_preserving():
    f = rand_flow(4, 3, additive=True, actnorm=False)
    _, logdet, _ = f.forward(np.random.default_rng(0).standard_normal((10, 4)))
    assert np.allclose(logdet, 0.0)


@pytest.mark.parametrize("additive", [False, True])
@pytest.mark.parametrize("d", [2, 5])
def test_gradients_match_finite_differences(additive, d):
    f = rand_flow(d, 3, additive=additive)
    x = np.random.default_rng(4).standard_normal((16, d))
    _, grads = f.loss_and_grads(x)
    grads = {k: g.copy() for k, g in grads.items()}
    params = f.named_params()
    rng = np.random.default_rng(5)
    checked = 0
    for name, p in params.items():
        flat = p.reshape(-1)
        for j in rng.choice(flat.size, size=min(3, flat.size), replace=False):
            old = flat[j]
            h = 1e-6
            flat[j] = old + h; lp = f.loss_and_grads(x)[0]
            flat[j] = old - h; lm = f.loss_and_grads(x)[0]
            flat[j] = old
            num = (lp - lm) / (2 * h)
            assert grads[name].reshape(-1)[j] == pytest.approx(num, rel=1e-4, abs=1e-7), name
            checked += 1
    assert checked > 20


def test_broken_backward_is_caught():
    """The finite-difference check has teeth: dropping the logdet term from the coupling backward fails it."""
    f = rand_flow(2, 2)
    x = np.random.default_rng(4).standard_normal((16, 2))
    c = [l for l in f.layers if isinstance(l, Coupling)][0]
    orig = c.backward
    c.backward = lambda xin, gy, gld: orig(xin, gy, np.zeros_like(gld))
    _, grads = f.loss_and_grads(x)
    p = c.params["W2"]
    old = p[0, 0]
    h = 1e-6
    p[0, 0] = old + h; lp = f.loss_and_grads(x)[0]
    p[0, 0] = old - h; lm = f.loss_and_grads(x)[0]
    p[0, 0] = old
    num = (lp - lm) / (2 * h)
    assert abs(grads["%d.W2" % f.layers.index(c)][0, 0] - num) > 1e-4


def test_zero_init_is_identity_up_to_actnorm():
    f = Flow(2, 4, 8, seed=0, actnorm=False)
    x = np.random.default_rng(0).standard_normal((20, 2))
    z, logdet, _ = f.forward(x)
    assert np.allclose(np.sort(z, axis=1), np.sort(x, axis=1)) and np.allclose(logdet, 0)


def test_actnorm_data_init_standardises():
    x = np.random.default_rng(0).standard_normal((4000, 3)) * [1, 5, 0.2] + [3, -2, 10]
    a = ActNorm(3)
    a.data_init(x)
    y, ld = a.forward(x)
    assert np.allclose(y.mean(0), 0, atol=1e-9) and np.allclose(y.std(0), 1, atol=1e-6)
    assert ld[0] == pytest.approx(-np.log(x.std(0)).sum(), abs=1e-4)


def test_permutation_roundtrip_and_backward():
    p = Permutation(np.array([2, 0, 1]))
    x = np.arange(12.0).reshape(4, 3)
    y, _ = p.forward(x)
    assert np.array_equal(p.inverse(y), x)
    g = np.random.default_rng(0).standard_normal((4, 3))
    assert np.allclose(p.backward(x, g, None)[:, 2], g[:, 0])  # y0 = x2


def test_untrained_flow_is_a_normalised_density():
    f = rand_flow(2, 3, w2=0.15)
    assert density_grid_integral(f, lim=10.0, res=600) == pytest.approx(1.0, abs=0.03)


def test_gaussian_data_nll_matches_closed_form():
    """Fit a pure ActNorm (a diagonal affine map) to Gaussian data: NLL -> the Gaussian's entropy."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((20000, 2)) * [2.0, 0.5] + [1, -1]
    f = Flow(2, 0, 4, seed=0)
    f.data_init(x)
    entropy = 0.5 * 2 * (1 + np.log(2 * np.pi)) + np.log(2.0) + np.log(0.5)
    assert f.nll(x) == pytest.approx(entropy, abs=0.02)


def test_fit_reduces_nll_and_beats_actnorm_only():
    ds = two_moons()
    x, xv = ds.sample(3000, 0), ds.sample(3000, 1)
    base = Flow(2, 0, 4, seed=0)
    base.data_init(x)
    f = Flow(2, 4, 32, seed=0)
    res = fit(f, x, steps=500, seed=0, x_val=xv)
    assert res.val[-1][1] < res.val[0][1]
    assert f.nll(xv) < base.nll(xv) - 0.3


def test_sample_inverts_density_roundtrip():
    f = rand_flow(2, 3)
    s = f.sample(100, seed=1)
    z, _, _ = f.forward(s)
    assert np.allclose(z, np.random.default_rng(1).standard_normal((100, 2)))


def test_ring_density_is_normalised_and_matches_sampling():
    ds = gaussian_ring()
    g = np.linspace(-4, 4, 700)
    xx, yy = np.meshgrid(g, g)
    p = np.exp(ds.log_prob(np.stack([xx.ravel(), yy.ravel()], 1)))
    assert p.sum() * (g[1] - g[0]) ** 2 == pytest.approx(1.0, abs=0.01)
    # entropy via sampling equals entropy via the grid
    x = ds.sample(200000, 0)
    mc = -ds.log_prob(x).mean()
    lp = np.log(np.maximum(p, 1e-300))
    grid = -(p * lp).sum() * (g[1] - g[0]) ** 2
    assert mc == pytest.approx(grid, abs=0.02)


def test_checkerboard_density_normalised_and_samples_inside():
    ds = checkerboard(4)
    x = ds.sample(5000, 0)
    assert np.isfinite(ds.log_prob(x)).all()
    g = np.linspace(-2, 2, 801)[:-1] + 0.0025
    xx, yy = np.meshgrid(g, g)
    p = np.exp(ds.log_prob(np.stack([xx.ravel(), yy.ravel()], 1)))
    assert p.sum() * 0.005 ** 2 == pytest.approx(1.0, abs=0.01)


def test_datasets_are_deterministic_and_shaped():
    for ds in (gaussian_ring(), two_moons(), checkerboard(), spiral()):
        a, b = ds.sample(100, 3), ds.sample(100, 3)
        assert a.shape == (100, 2) and np.array_equal(a, b)
        assert not np.array_equal(a, ds.sample(100, 4))


def test_bridge_mass_counts_off_mode_samples():
    ds = gaussian_ring()
    assert bridge_mass(ds.sample(5000, 0), ds.centers, ds.sigma, 5.0) < 0.001
    assert bridge_mass(np.zeros((10, 2)), ds.centers, ds.sigma) == 1.0


def test_nll_gap_is_near_zero_for_the_true_density_proxy():
    """A flow trained on the ring should have a non-negative gap (up to MC noise), never a large negative one."""
    ds = gaussian_ring()
    x = ds.sample(4000, 0)
    f = Flow(2, 4, 32, seed=0)
    fit(f, x, steps=300, seed=0)
    r = nll_gap(f, ds, n=5000)
    assert r["gap"] > -0.05
    with pytest.raises(ValueError):
        nll_gap(f, two_moons())


def test_validation():
    with pytest.raises(ValueError):
        Coupling(1, 4)


def test_ascii_density_shape_and_content():
    from normflow import ascii_density
    pts = np.array([[0.0, 0.0]] * 50 + [[2.0, 2.0]])
    s = ascii_density(pts, lim=3.0, width=20, height=10).split("\n")
    assert len(s) == 10 and all(len(r) == 20 for r in s)
    assert s[5][10] == "@" or s[4][10] == "@" or "@" in "".join(s)       # the heavy cell hits the top of the ramp
    assert sum(c != " " for r in s for c in r) == 2                       # two occupied cells
    assert "@" not in ascii_density(np.zeros((0, 2)) + 99, 3.0, 20, 10)   # out-of-range points are ignored
    # y grows upward: a point at the top of the window lands in the first row
    assert ascii_density(np.array([[0.0, 2.9]]), 3.0, 20, 10).split("\n")[0].strip() != ""


def test_invlinear_grad_and_logdet():
    from normflow import InvLinear
    rng = np.random.default_rng(0)
    l = InvLinear(3, seed=1)
    l.params["W"] = l.params["W"] * 1.3 + 0.2 * rng.standard_normal((3, 3))
    x = rng.standard_normal((5, 3)); gy = rng.standard_normal((5, 3)); gl = rng.standard_normal(5)
    y, ld = l.forward(x)
    assert np.allclose(l.inverse(y), x)
    assert ld[0] == pytest.approx(np.log(abs(np.linalg.det(l.params["W"]))))
    def f():
        y, ld = l.forward(x)
        return float((y * gy).sum() + (ld * gl).sum())
    l.grads["W"][...] = 0
    gx = l.backward(x, gy, gl)
    W = l.params["W"]
    for i in range(3):
        for j in range(3):
            W[i, j] += 1e-6; a = f(); W[i, j] -= 2e-6; b = f(); W[i, j] += 1e-6
            assert l.grads["W"][i, j] == pytest.approx((a - b) / 2e-6, rel=1e-5, abs=1e-6)
    assert gx[0, 0] == pytest.approx(gy[0] @ W[0], rel=1e-9)


def test_linear_mixing_flow_roundtrip_and_trains():
    f = Flow(2, 3, 16, seed=0, mixing="linear")
    x = two_moons().sample(600, seed=0)
    res = fit(f, x, steps=150, batch=128, seed=0)
    z, _, _ = f.forward(x)
    assert np.allclose(f.inverse(z), x, atol=1e-7)
    assert res.losses[-1][1] < res.losses[0][1]
    with pytest.raises(ValueError):
        Flow(2, mixing="bogus")


def test_broken_invlinear_backward_fails():
    from normflow import InvLinear
    l = InvLinear(3, seed=0)
    x = np.random.default_rng(0).standard_normal((4, 3))
    l.grads["W"][...] = 0
    l.backward(x, np.zeros((4, 3)), np.ones(4))
    # with only a logdet signal the gradient must be 4*W^{-T}; zero would be a broken implementation
    assert np.allclose(l.grads["W"], 4 * np.linalg.inv(l.params["W"]).T)
