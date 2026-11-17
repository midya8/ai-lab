import numpy as np
import pytest

from diffusion2d import (EpsNet, GMM, cosine_schedule, linear_schedule, mmd_rbf, net_eps_fn,
                          oracle_eps_fn, q_sample, ring, sample, sliced_wasserstein, timestep_grid, train)
from diffusion2d.diagnostics import ascii_scatter, eps_error_at


# ---------------------------------------------------------------- schedules
@pytest.mark.parametrize("make", [linear_schedule, cosine_schedule])
def test_schedule_is_valid(make):
    s = make(100)
    assert s.abar[0] == 1.0 and s.T == 100
    assert np.all(np.diff(s.abar) < 0) and np.all(s.abar[1:] > 0)
    assert s.abar[-1] < 1e-2            # almost pure noise at t = T
    assert np.all(s.betas[1:] > 0) and np.all(s.betas < 1)


def test_linear_schedule_total_noise_is_independent_of_T():
    la, lb = -np.log(linear_schedule(100).abar[-1]), -np.log(linear_schedule(400).abar[-1])
    assert la == pytest.approx(lb, rel=0.1)   # same total log-SNR range up to discretisation


def test_q_sample_moments():
    s = linear_schedule(100)
    rng = np.random.default_rng(0)
    x0 = np.full((200_000, 2), 3.0)
    t = np.full(len(x0), 40)
    xt = q_sample(x0, t, rng.standard_normal(x0.shape), s)
    assert xt.mean() == pytest.approx(np.sqrt(s.abar[40]) * 3.0, abs=0.01)
    assert xt.var() == pytest.approx(1 - s.abar[40], abs=0.01)


# ---------------------------------------------------------------- exact GMM maths
def test_exact_score_matches_finite_difference_of_log_density():
    g = ring(5, 1.5, 0.3)
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.5, (20, 2))
    for ab in (1.0, 0.7, 0.2, 0.01):
        sc = g.score(x, ab)
        h = 1e-5
        fd = np.zeros_like(x)
        for d in range(2):
            e = np.zeros(2); e[d] = h
            fd[:, d] = (g.log_density(x + e, ab) - g.log_density(x - e, ab)) / (2 * h)
        np.testing.assert_allclose(sc, fd, atol=1e-5, rtol=1e-5)


def test_noised_marginal_matches_forward_process_samples():
    """Mean/covariance of q(x_t) samples equal the mixture's closed form."""
    g = ring(6, 1.0, 0.4)
    s = linear_schedule(100)
    rng = np.random.default_rng(0)
    x0 = g.sample(300_000, rng)
    t = 30
    xt = q_sample(x0, np.full(len(x0), t), rng.standard_normal(x0.shape), s)
    ab = s.abar[t]
    # symmetric ring: zero mean; covariance = ab*(radius^2/2 + std^2) + (1-ab) on the diagonal
    assert np.abs(xt.mean(0)).max() < 0.02
    expected_var = ab * (1.0 ** 2 / 2 + 0.4 ** 2) + (1 - ab)
    assert np.diag(np.cov(xt.T)) == pytest.approx([expected_var] * 2, abs=0.02)


def test_eps_star_is_the_conditional_mean_of_the_noise():
    """eps* minimises squared error: any perturbation increases MSE, and the loss
    decomposes into excess + irreducible."""
    g = ring(8, 1.5, 0.3)
    s = linear_schedule(100)
    rng = np.random.default_rng(3)
    t = 35
    x0 = g.sample(200_000, rng)
    eps = rng.standard_normal(x0.shape)
    xt = q_sample(x0, np.full(len(x0), t), eps, s)
    star = g.eps_star(xt, s.abar[t])
    base = ((star - eps) ** 2).mean()
    pert = star + 0.1 * np.sin(3 * xt)
    assert ((pert - eps) ** 2).mean() > base
    # Pythagoras: E|p-eps|^2 = E|p-eps*|^2 + E|eps*-eps|^2
    lhs = ((pert - eps) ** 2).mean()
    rhs = ((pert - star) ** 2).mean() + base
    assert lhs == pytest.approx(rhs, rel=5e-3)


def test_class_conditional_eps_matches_class_score():
    g = ring(4, 1.5, 0.3)
    x = np.random.default_rng(0).normal(size=(10, 2))
    ab = 0.4
    lab = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    got = g.eps_star_class(x, ab, lab)
    want = np.stack([-np.sqrt(1 - ab) * g.class_score(x[i:i + 1], ab, lab[i])[0] for i in range(10)])
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_mode_stats_perfect_and_collapsed():
    g = ring(8, 1.5, 0.3)
    x = g.sample(20_000, np.random.default_rng(0))
    st = g.mode_stats(x)
    assert st["on_mode"] > 0.99 and st["mode_tv"] < 0.03
    collapsed = g.means[np.zeros(1000, int)] + 0.01
    assert g.mode_stats(collapsed)["mode_tv"] == pytest.approx(1 - 1 / 8, abs=1e-9)


# ---------------------------------------------------------------- network
def _grad_check(net, labels=None, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(6, 2)); t = rng.integers(1, 100, 6); target = rng.normal(size=(6, 2))
    for p in net.params:                     # randomise so no gradient is trivially zero
        p += rng.normal(0, 0.1, p.shape)
    _, grads = net.loss_and_grads(x, t, labels, target)
    h = 1e-6
    worst = 0.0
    for p, g in zip(net.params, grads):
        flat, gflat = p.reshape(-1), g.reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + h; lp = net.loss_and_grads(x, t, labels, target)[0]
            flat[i] = old - h; lm = net.loss_and_grads(x, t, labels, target)[0]
            flat[i] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - gflat[i]) / (abs(num) + abs(gflat[i]) + 1e-8))
    return worst


def test_net_gradcheck_unconditional():
    assert _grad_check(EpsNet(2, 16, 2, 0, T=100)) < 1e-5


def test_net_gradcheck_conditional_with_null_labels():
    net = EpsNet(2, 16, 3, 4, T=100)
    assert _grad_check(net, np.array([0, 1, -1, 3, -1, 2])) < 1e-5


def test_null_label_equals_missing_labels():
    net = EpsNet(2, 16, 2, 3, T=100)
    for p in net.params:
        p += 0.1
    net.ema = [p.copy() for p in net.params]
    x = np.random.default_rng(0).normal(size=(5, 2)); t = np.full(5, 20)
    np.testing.assert_array_equal(net.predict(x, t, np.full(5, -1)), net.predict(x, t, None))
    assert not np.allclose(net.predict(x, t, np.zeros(5, int)), net.predict(x, t, None))


def test_training_reduces_loss_and_gets_near_optimal_at_high_noise():
    g, s = ring(), linear_schedule(100)
    net = EpsNet(2, 64, 2, 0, T=100)
    h = train(net, lambda n, r: (g.sample(n, r), None), s, steps=600, batch=256, lr=3e-3, p_uncond=0)
    assert np.mean(h[-50:]) < 0.8 * np.mean(h[:50])
    r = eps_error_at(net, g, s, 80, n=4000)
    assert r["excess"] < 0.05
    # loss = excess + irreducible (Monte-Carlo identity)
    assert r["loss"] == pytest.approx(r["excess"] + r["irreducible"], abs=0.01)


# ---------------------------------------------------------------- samplers
def test_timestep_grid():
    g = timestep_grid(100, 10)
    assert g[0] == 100 and g[-1] == 0 and len(g) == 11 and np.all(np.diff(g) < 0)
    assert list(timestep_grid(100, 100)) == list(range(100, -1, -1))
    assert len(timestep_grid(10, 50)) == 11          # cannot take more steps than T


def test_ddim_eta1_full_steps_matches_ddpm_posterior():
    """One generalised-DDIM step with eta=1 has Ho et al.'s mean and variance."""
    s = linear_schedule(100)
    t = 57
    x = np.random.default_rng(0).normal(size=(3, 2))
    eps = np.random.default_rng(1).normal(size=(3, 2))
    mean_ddpm = (x - s.betas[t] / np.sqrt(1 - s.abar[t]) * eps) / np.sqrt(s.alphas[t])
    # deterministic part of the DDIM update at eta=1
    ab_t, ab_s = s.abar[t], s.abar[t - 1]
    sigma = np.sqrt((1 - ab_s) / (1 - ab_t)) * np.sqrt(1 - ab_t / ab_s)
    x0 = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
    mean_ddim = np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s - sigma ** 2) * eps
    np.testing.assert_allclose(mean_ddim, mean_ddpm, atol=1e-9)
    assert sigma ** 2 == pytest.approx(s.posterior_variance(t), rel=1e-9)


def test_ddim_is_deterministic_given_x_T_and_independent_of_rng():
    g, s = ring(), linear_schedule(100)
    xT = np.random.default_rng(5).standard_normal((50, 2))
    a = sample(oracle_eps_fn(g, s), s, 50, 2, 20, 0.0, np.random.default_rng(1), x_T=xT)
    b = sample(oracle_eps_fn(g, s), s, 50, 2, 20, 0.0, np.random.default_rng(2), x_T=xT)
    np.testing.assert_array_equal(a, b)


def test_oracle_sampler_recovers_the_distribution():
    g, s = ring(), linear_schedule(100)
    ref = g.sample(4000, np.random.default_rng(1))
    floor = sliced_wasserstein(g.sample(4000, np.random.default_rng(2)), ref)
    for steps, eta in [(100, 1.0), (50, 0.0)]:
        x = sample(oracle_eps_fn(g, s), s, 4000, 2, steps, eta, np.random.default_rng(3))
        assert sliced_wasserstein(x, ref) < floor + 0.03
        assert g.mode_stats(x)["on_mode"] > 0.97


def test_single_step_oracle_ddim_collapses_towards_the_mean():
    """One DDIM step maps noise to ~E[x0]: sample variance is far below the data's."""
    g, s = ring(), linear_schedule(100)
    x = sample(oracle_eps_fn(g, s), s, 4000, 2, 1, 0.0, np.random.default_rng(0))
    assert x.std() < 0.5 * g.sample(4000, np.random.default_rng(0)).std()


def test_guidance_scale_one_is_conditional_and_zero_is_unconditional():
    net = EpsNet(2, 16, 2, 3, T=100)
    for p in net.params:
        p += np.random.default_rng(0).normal(0, 0.2, p.shape)
    net.ema = [p.copy() for p in net.params]
    x = np.random.default_rng(1).normal(size=(4, 2)); lab = np.array([0, 1, 2, 0])
    c = net.predict(x, np.full(4, 30), lab); u = net.predict(x, np.full(4, 30), None)
    np.testing.assert_allclose(net_eps_fn(net, 1.0)(x, 30, lab), c)
    np.testing.assert_allclose(net_eps_fn(net, 0.0)(x, 30, lab), u)
    np.testing.assert_allclose(net_eps_fn(net, 3.0)(x, 30, lab), u + 3 * (c - u))


def test_oracle_guidance_increases_class_accuracy_and_shrinks_spread():
    g, s = ring(8, 1.5, 0.3), linear_schedule(100)
    cond, unc = oracle_eps_fn(g, s), oracle_eps_fn(g, s)
    lab = np.full(2000, 2)
    res = {}
    for w in (1.0, 4.0):
        fn = lambda x, t, l, w=w: unc(x, t, None) + w * (cond(x, t, l) - unc(x, t, None))
        x = sample(fn, s, 2000, 2, 50, 0.0, np.random.default_rng(0), labels=lab)
        res[w] = ((g.assign(x) == 2).mean(), x.std(0).mean())
    assert res[4.0][0] >= res[1.0][0] - 0.01
    assert res[4.0][1] < res[1.0][1]


# ---------------------------------------------------------------- metrics
def test_sliced_wasserstein_properties():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 2))
    assert sliced_wasserstein(a, a) == 0.0
    # shifting by delta along one axis: mean |delta * u_x| over random unit u = delta * 2/pi in 2-D
    shifted = a + np.array([1.0, 0.0])
    assert sliced_wasserstein(a, shifted, n_proj=2000) == pytest.approx(2 / np.pi, abs=0.03)
    assert sliced_wasserstein(a, rng.normal(size=(2000, 2))) < 0.1


def test_mmd_separates_distributions():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(600, 2)), rng.normal(size=(600, 2))
    c = rng.normal(loc=1.0, size=(600, 2))
    assert abs(mmd_rbf(a, b)) < 0.02
    assert mmd_rbf(a, c) > 0.1


def test_ascii_scatter_shape():
    out = ascii_scatter(np.random.default_rng(0).normal(size=(500, 2)), width=30, height=10).split("\n")
    assert len(out) == 10 and all(len(r) == 30 for r in out)
