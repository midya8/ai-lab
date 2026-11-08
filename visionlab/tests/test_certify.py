import numpy as np
import pytest

from visionlab.attacks import pgd
from visionlab.certify import (certified_accuracy, certified_mask, certified_radius, ibp_loss_backward,
                               input_box, logit_bounds, margin_lower_bounds)
from visionlab.data import make_dataset
from visionlab.layers import Dense
from visionlab.model import ModelConfig, Sequential, build_cnn
from visionlab.train import train

from test_layers import num_grad


def tiny(seed=0, **kw):
    """Small float64 net with randomised biases / BN statistics so no layer is trivially the identity."""
    m = build_cnn(ModelConfig(img=8, width=3, hidden=8, n_classes=3, dtype="float64", seed=seed, **kw))
    rng = np.random.default_rng(seed + 100)
    for layer in m.layers:
        if hasattr(layer, "b"):
            layer.b[...] = rng.standard_normal(layer.b.shape) * 0.2
        if hasattr(layer, "running_mean"):
            layer.running_mean[...] = rng.standard_normal(layer.ch) * 0.1
            layer.running_var[...] = rng.uniform(0.5, 1.5, layer.ch)
            layer.gamma[...] = rng.uniform(-1.3, 1.3, layer.ch)     # negative slopes exercise |s|
            layer.beta[...] = rng.standard_normal(layer.ch) * 0.1
    return m.eval()


def images(n=6, seed=1):
    return np.random.default_rng(seed).random((n, 1, 8, 8))


ARCHS = [dict(), dict(head="gap"), dict(head="gap", batchnorm=True), dict(batchnorm=True)]


def test_zero_radius_bounds_equal_the_logits():
    for kw in ARCHS:
        m, x = tiny(**kw), images()
        lo, hi = logit_bounds(m, x, 0.0)
        z = m.predict_logits(x)
        np.testing.assert_allclose(lo, z, atol=1e-10)
        np.testing.assert_allclose(hi, z, atol=1e-10)


def test_bounds_enclose_random_and_adversarial_points_in_the_box():
    rng = np.random.default_rng(3)
    eps = 0.08
    for kw in ARCHS:
        m, x = tiny(**kw), images()
        y = m.predict(x)
        lo, hi = logit_bounds(m, x, eps)
        pts = [np.clip(x + rng.uniform(-eps, eps, x.shape), 0, 1) for _ in range(40)]
        pts += [np.clip(x + eps * rng.choice([-1.0, 1.0], x.shape), 0, 1) for _ in range(40)]   # box corners
        pts.append(pgd(m, x, y, eps, steps=8, rng=rng).astype(np.float64))
        for p in pts:
            z = m.predict_logits(p)
            assert (z >= lo - 1e-9).all() and (z <= hi + 1e-9).all(), kw


def test_bounds_widen_monotonically_with_eps():
    m, x = tiny(head="gap", batchnorm=True), images()
    prev_lo, prev_hi = logit_bounds(m, x, 0.01)
    for eps in (0.03, 0.1, 0.3):
        lo, hi = logit_bounds(m, x, eps)
        assert (lo <= prev_lo + 1e-12).all() and (hi >= prev_hi - 1e-12).all()
        prev_lo, prev_hi = lo, hi


def test_input_box_is_clipped_to_the_pixel_range_and_accepts_per_image_eps():
    x = np.array([[[[0.0, 0.5, 1.0]]]])
    mu, r = input_box(x, 0.2)
    np.testing.assert_allclose(mu - r, [[[[0.0, 0.3, 0.8]]]])
    np.testing.assert_allclose(mu + r, [[[[0.2, 0.7, 1.0]]]])
    x2 = np.full((2, 1, 1, 2), 0.5)
    mu, r = input_box(x2, np.array([0.1, 0.3]))
    np.testing.assert_allclose(r[:, 0, 0, 0], [0.1, 0.3])


def test_a_single_linear_layer_is_bounded_exactly():
    rng = np.random.default_rng(5)
    d = Dense(5, 3, rng, np.float64)
    d.b[...] = rng.standard_normal(3)
    m = Sequential([d])
    x = rng.random((4, 5)) * 0.6 + 0.2
    lo, hi = logit_bounds(m, x, 0.1)
    # extreme points of the box are attained by picking each coordinate's endpoint by the weight sign
    for j in range(3):
        best = np.where(d.W[:, j] > 0, x + 0.1, x - 0.1) @ d.W[:, j] + d.b[j]
        worst = np.where(d.W[:, j] > 0, x - 0.1, x + 0.1) @ d.W[:, j] + d.b[j]
        np.testing.assert_allclose(hi[:, j], best)
        np.testing.assert_allclose(lo[:, j], worst)


def test_margin_bound_is_sound_and_at_least_as_tight_as_naive_interval_comparison():
    for kw in ARCHS:
        m, x = tiny(**kw), images(12)
        y = m.predict(x)
        for eps in (0.01, 0.03, 0.1):
            marg = margin_lower_bounds(m, x, y, eps)
            z = m.predict_logits(x)
            actual = z[np.arange(len(y)), y][:, None] - z
            assert (marg[np.isfinite(marg)] <= actual[np.isfinite(marg)] + 1e-9).all()
            lo, hi = logit_bounds(m, x, eps)
            hi = hi.copy()
            hi[np.arange(len(y)), y] = -np.inf
            naive_margin = lo[np.arange(len(y)), y][:, None] - hi
            naive_margin[np.arange(len(y)), y] = np.inf
            assert (marg >= naive_margin - 1e-9).all()
            assert (certified_mask(m, x, y, eps, "naive") <= certified_mask(m, x, y, eps, "margin")).all()


_cache = {}


def trained_pair():
    """(standard, IBP-trained) default-architecture models on the same 2000 images (cached)."""
    if "p" not in _cache:
        Xtr, ytr = make_dataset(2000, seed=1)
        std = build_cnn(ModelConfig())
        train(std, Xtr, ytr, epochs=8, log=None)
        ibp = build_cnn(ModelConfig())
        train(ibp, Xtr, ytr, epochs=8, log=None, ibp_eps=0.03)
        _cache["p"] = (std, ibp)
    return _cache["p"]


def test_certified_images_survive_pgd():
    """Soundness at the decision level: no certified image can be flipped by an attack inside the box."""
    rng = np.random.default_rng(8)
    X, y = make_dataset(150, seed=4)
    _, m = trained_pair()
    eps = 0.02
    cert = certified_mask(m, X, y, eps)
    assert cert.sum() > 0
    adv = pgd(m, X, y, eps, steps=20, rng=rng)
    assert (m.predict(adv)[cert] == y[cert]).all()
    assert (m.predict(X)[cert] == y[cert]).all()


def test_certified_accuracy_falls_with_eps_and_equals_clean_accuracy_at_zero():
    m = tiny(head="gap")
    x = images(30)
    y = m.predict(x)                       # every image is "correct" by construction
    accs = [certified_accuracy(m, x, y, e) for e in (0.0, 0.01, 0.05, 0.2, 0.6)]
    assert accs[0] == 1.0
    assert all(a >= b for a, b in zip(accs, accs[1:]))
    assert accs[-1] < 1.0


def test_certified_radius_brackets_the_certification_boundary():
    m = tiny(head="gap", batchnorm=True)
    x = images(8)
    y = m.predict(x)
    rad = certified_radius(m, x, y, hi=1.0, iters=16)
    res = 1.0 / 2 ** 16
    for i in range(len(x)):
        xi, yi = x[i:i + 1], y[i:i + 1]
        if rad[i] > 0:
            assert certified_mask(m, xi, yi, float(rad[i]))[0]
        if rad[i] < 1.0 - 2 * res:
            assert not certified_mask(m, xi, yi, float(rad[i]) + 2 * res)[0]
    wrong = (y + 1) % 3
    assert (certified_radius(m, x, wrong, hi=1.0, iters=6) == 0).all()


def test_ibp_loss_gradient_matches_finite_differences_for_every_parameter():
    rng = np.random.default_rng(11)
    for kw in (dict(), dict(head="gap")):
        m = tiny(**kw)
        x, y = rng.random((4, 1, 8, 8)), np.array([0, 1, 2, 1])
        eps = 0.05
        f = lambda: ibp_loss_backward(m, x, y, eps)[0]
        ibp_loss_backward(m, x, y, eps)
        for name, p, g in m.parameters():
            np.testing.assert_allclose(g, num_grad(f, p), atol=1e-6, rtol=1e-4, err_msg=f"{kw} {name}")


def test_ibp_loss_at_zero_eps_is_plain_cross_entropy_with_the_same_gradient():
    from visionlab.model import cross_entropy
    m = tiny()
    rng = np.random.default_rng(12)
    x, y = rng.random((5, 1, 8, 8)), np.array([0, 1, 2, 1, 0])
    loss, dl = cross_entropy(m.forward(x), y)
    m.backward(dl)
    ref = [g.copy() for _, _, g in m.parameters()]
    iloss, _ = ibp_loss_backward(m, x, y, 0.0)
    assert iloss == pytest.approx(loss)
    for (_, _, g), g0 in zip(m.parameters(), ref):
        np.testing.assert_allclose(g, g0, atol=1e-10)


def test_worst_case_loss_never_below_clean_loss():
    m = tiny()
    rng = np.random.default_rng(13)
    x, y = rng.random((8, 1, 8, 8)), rng.integers(0, 3, 8)
    base = ibp_loss_backward(m, x, y, 0.0)[0]
    assert ibp_loss_backward(m, x, y, 0.05)[0] >= base - 1e-12


def test_ibp_training_certifies_more_than_standard_training():
    std, ibp = trained_pair()
    Xte, yte = make_dataset(300, seed=2)
    eps = 0.03
    c_std, c_ibp = certified_accuracy(std, Xte, yte, eps), certified_accuracy(ibp, Xte, yte, eps)
    assert c_ibp > c_std + 0.1
    assert certified_accuracy(ibp, Xte, yte, 0.0) > 0.25     # short run: clean accuracy is low (chance is 1/6)


def test_ibp_training_rejects_unsupported_combinations():
    X, y = make_dataset(64, seed=1)
    with pytest.raises(ValueError):
        train(build_cnn(ModelConfig(head="gap", batchnorm=True)), X, y, epochs=1, log=None, ibp_eps=0.02)
    with pytest.raises(ValueError):
        train(build_cnn(ModelConfig()), X, y, epochs=1, log=None, ibp_eps=0.02, adv_eps=0.05)


def test_unknown_layer_type_is_reported():
    class Weird:
        training = False

        def params(self):
            return {}

    from visionlab.certify import _make_op
    with pytest.raises(TypeError):
        _make_op(Weird())
