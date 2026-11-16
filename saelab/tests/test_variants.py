import numpy as np
import pytest

from saelab import (GatedSAE, JumpReLUSAE, absorption, fit, fvu, make_hierarchical_world, make_world,
                    feature_recovery)


def _fd(loss_fn, grads, params, skip=(), n_probe=10, eps=1e-6, tol=1e-5, seed=0):
    rng = np.random.default_rng(seed)
    for name, p in params.items():
        if name in skip:
            continue
        flat = p.reshape(-1)
        for i in rng.choice(flat.size, size=min(n_probe, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = loss_fn()
            flat[i] = old - eps
            lm = loss_fn()
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            g = grads[name].reshape(-1)[i]
            assert abs(fd - g) <= tol + 1e-4 * abs(fd), (name, i, fd, g)


def _jr(seed=3, **kw):
    rng = np.random.default_rng(seed)
    sae = JumpReLUSAE(6, 14, seed=1, theta_init=0.05, **kw)
    sae.params["b_enc"] += rng.normal(0, 0.1, 14)
    sae.params["b_dec"] += rng.normal(0, 0.1, 6)
    return sae, rng.normal(size=(40, 6))


def test_jumprelu_gradients_match_finite_differences_away_from_thresholds():
    sae, x = _jr(l0_coef=0.05)
    _, grads, info = sae.loss_and_grads(x)
    assert 0 < info["active"] < 14        # a non-trivial active set, otherwise the check is vacuous
    _fd(lambda: sae.loss_and_grads(x)[0], grads, sae.params, skip=("theta",))


def test_broken_jumprelu_gradient_is_caught():
    sae, x = _jr()
    _, grads, _ = sae.loss_and_grads(x)
    p, eps = sae.params["W_dec"], 1e-6
    worst = []
    for scale in (1.0, 1.4):
        w = 0.0
        for i, j in [(0, 0), (3, 2), (5, 4), (9, 1), (12, 5)]:
            old = p[i, j]
            p[i, j] = old + eps
            lp = sae.loss_and_grads(x)[0]
            p[i, j] = old - eps
            lm = sae.loss_and_grads(x)[0]
            p[i, j] = old
            w = max(w, abs((lp - lm) / (2 * eps) - scale * grads["W_dec"][i, j]))
        worst.append(w)
    assert worst[0] < 1e-6 and worst[1] > 1e-4


def test_jumprelu_theta_pseudo_gradient_matches_a_reference_loop():
    sae, x = _jr(l0_coef=0.07, eps=0.4)
    p = sae.params
    _, grads, _ = sae.loss_and_grads(x)
    n = x.shape[0]
    pre = (x - p["b_dec"]) @ p["W_enc"] + p["b_enc"]
    err = (pre * (pre > p["theta"])) @ p["W_dec"] + p["b_dec"] - x
    da = (2 * err / n) @ p["W_dec"].T
    ref = np.zeros(sae.m)
    for j in range(sae.m):
        for i in range(n):
            u = (pre[i, j] - p["theta"][j]) / sae.eps
            if abs(u) < 0.5:
                ref[j] += -da[i, j] * p["theta"][j] / sae.eps - sae.l0_coef / (n * sae.eps)
    assert np.allclose(grads["theta"], ref)
    assert np.abs(ref).max() > 0          # the kernel really overlaps some samples


def test_jumprelu_zeroes_below_threshold_and_keeps_value_above():
    sae = JumpReLUSAE(4, 6, seed=0)
    sae.params["theta"][:] = np.array([0.1, 0.5, 1.0, 2.0, 0.3, 0.2])
    x = np.random.default_rng(0).normal(size=(100, 4))
    pre = (x - sae.params["b_dec"]) @ sae.params["W_enc"] + sae.params["b_enc"]
    a = sae.encode(x)
    assert np.array_equal(a > 0, pre > sae.params["theta"])
    assert np.allclose(a[a > 0], pre[a > 0])       # no shrinkage: the value above theta is untouched


def test_larger_l0_coefficient_pushes_thresholds_up():
    w = make_world(16, 16, l0=2, orthogonal=True, seed=0)
    th = []
    for c in (0.001, 0.1):
        sae = JumpReLUSAE(16, 32, l0_coef=c, seed=0)
        fit(sae, w, steps=400, batch=256)
        th.append(float(np.median(sae.params["theta"])))
    assert th[1] > th[0]


def test_normalize_decoder_preserves_function_for_both_variants():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(30, 6))
    for sae in (JumpReLUSAE(6, 12, seed=0, theta_init=0.1), GatedSAE(6, 12, seed=0)):
        sae.params["W_dec"] *= rng.uniform(0.5, 3.0, size=(12, 1))
        if isinstance(sae, JumpReLUSAE):
            sae.params["W_enc"] /= np.linalg.norm(sae.params["W_dec"], axis=1)[None, :]
            sae.params["b_enc"] /= np.linalg.norm(sae.params["W_dec"], axis=1)
            sae.params["theta"] /= np.linalg.norm(sae.params["W_dec"], axis=1)
        else:
            sae.params["r_mag"] -= np.log(np.linalg.norm(sae.params["W_dec"], axis=1))
        before = sae.reconstruct(x)
        sae.normalize_decoder()
        assert np.allclose(np.linalg.norm(sae.params["W_dec"], axis=1), 1.0)
        assert np.allclose(sae.reconstruct(x), before)


# ---------------------------------------------------------------- gated
def _gated(seed=3):
    rng = np.random.default_rng(seed)
    sae = GatedSAE(6, 14, l1=0.3, seed=1)
    for k in ("b_gate", "b_mag", "r_mag"):
        sae.params[k] += rng.normal(0, 0.1, 14)
    sae.params["b_dec"] += rng.normal(0, 0.1, 6)
    return sae, rng.normal(size=(40, 6))


def test_gated_gradients_match_finite_differences_with_stop_gradient_decoder():
    sae, x = _gated()
    aux_W, aux_b = sae.params["W_dec"].copy(), sae.params["b_dec"].copy()   # frozen copies, as the loss treats them
    _, grads, info = sae.loss_and_grads(x)
    assert 0 < info["active"] < 14
    _fd(lambda: sae._loss_value(x, aux_W, aux_b), grads, sae.params)


def test_gated_loss_matches_its_reference_value():
    sae, x = _gated()
    loss = sae.loss_and_grads(x)[0]
    assert loss == pytest.approx(sae._loss_value(x, sae.params["W_dec"], sae.params["b_dec"]))


def test_gated_magnitude_path_is_tied_to_gate_weights():
    sae, x = _gated()
    sae.params["r_mag"][:] = np.log(2.0)
    sae.params["b_mag"][:] = 0.0
    sae.params["b_gate"][:] = 0.0
    xc = x - sae.params["b_dec"]
    u = xc @ sae.params["W_gate"]
    expect = (u > 0) * np.maximum(2.0 * u, 0.0)       # magnitude = 2 * gate pre-activation
    assert np.allclose(sae.encode(x), expect)


def test_gated_aux_gradient_does_not_reach_the_decoder():
    # the auxiliary term must not move W_dec: compare with a model whose aux weight is irrelevant
    sae, x = _gated()
    _, grads, _ = sae.loss_and_grads(x)
    p = sae.params
    xc = x - p["b_dec"]
    u = xc @ p["W_gate"] + p["b_gate"]
    v = xc @ (p["W_gate"] * np.exp(p["r_mag"])) + p["b_mag"]
    f = (u > 0) * np.maximum(v, 0)
    only_main = f.T @ (2 * (f @ p["W_dec"] + p["b_dec"] - x) / x.shape[0])
    assert np.allclose(grads["W_dec"], only_main)


# ---------------------------------------------------------------- end to end
@pytest.mark.parametrize("make", [lambda: JumpReLUSAE(8, 16, l0_coef=0.1, seed=0),
                                  lambda: GatedSAE(8, 16, l1=0.8, seed=0)])
def test_variants_recover_an_orthogonal_toy_world(make):
    w = make_world(8, 8, l0=2, orthogonal=True, seed=0)
    sae = make()
    fit(sae, w, steps=1500, batch=256, lr=3e-3)
    x, _ = w.sample(2000, np.random.default_rng(5))
    assert fvu(x, sae.reconstruct(x)) < 0.2
    assert feature_recovery(w.directions, sae.params["W_dec"])["matched"] >= 0.75


# ---------------------------------------------------------------- hierarchy + absorption
def test_children_never_fire_without_their_parent_and_marginals_are_right():
    w = make_hierarchical_world(4, 3, d=16, p_parent=0.3, p_child=0.5, seed=0)
    _, a = w.sample(40000, np.random.default_rng(0))
    assert w.n_features == 16 and (w.parent[:4] == -1).all()
    for c in range(4, 16):
        assert not ((a[:, c] > 0) & (a[:, w.parent[c]] == 0)).any()
    assert (a[:, 0] > 0).mean() == pytest.approx(0.3, abs=0.01)
    assert (a[:, 4] > 0).mean() == pytest.approx(0.15, abs=0.01)
    assert w.expected_l0 == pytest.approx(4 * 0.3 + 12 * 0.15)


def test_child_cos_sets_the_parent_child_angle():
    w = make_hierarchical_world(3, 2, d=16, child_cos=0.6, seed=1)
    for c in range(3, 9):
        assert w.directions[c] @ w.directions[w.parent[c]] == pytest.approx(0.6)
    assert np.allclose(np.linalg.norm(w.directions, axis=1), 1.0)
    with pytest.raises(ValueError):
        make_hierarchical_world(2, 2, 8, child_cos=1.0)


def test_absorption_metric_on_constructed_cases():
    w = make_hierarchical_world(2, 1, d=8, p_parent=0.5, p_child=0.5, seed=0)
    rng = np.random.default_rng(0)
    x, a = w.sample(4000, rng)
    # a perfect "SAE": one latent per true feature, fires exactly when the feature does
    dec = w.directions
    assert absorption(w, a, a, dec)["absorption"] == 0.0
    # an absorbing SAE: the parent latent is silent whenever its child is active
    z = a.copy()
    for c in (2, 3):
        z[a[:, c] > 0, w.parent[c]] = 0.0
    r = absorption(w, z, a, dec)
    assert r["absorption"] == pytest.approx(1.0) and r["rate_no_child"] == pytest.approx(1.0)
    assert r["rate_with_child"] == 0.0 and r["parents_found"] == 1.0
    with pytest.raises(ValueError):
        absorption(make_world(4, 8), a, a, dec)


def test_absorption_skips_parents_without_a_matching_latent():
    w = make_hierarchical_world(2, 1, d=8, seed=0)
    _, a = w.sample(500, np.random.default_rng(0))
    r = absorption(w, a, a, np.random.default_rng(1).standard_normal((4, 8)), parent_thresh=0.99)
    assert r["parents_found"] == 0.0 and np.isnan(r["absorption"])
