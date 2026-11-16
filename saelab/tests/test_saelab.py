import numpy as np
import pytest

from saelab import (SAE, coefficient_ratio, dead_latents, feature_recall, feature_recovery, fit,
                    fvu, l0, latent_purity, make_world)
from saelab.train import resample_dead


# ---------------------------------------------------------------- data
def test_world_directions_unit_and_samples_nonnegative():
    w = make_world(40, 16, l0=3, seed=1)
    assert np.allclose(np.linalg.norm(w.directions, axis=1), 1.0)
    x, a = w.sample(500, np.random.default_rng(0))
    assert x.shape == (500, 16) and a.shape == (500, 40)
    assert (a >= 0).all()
    assert np.allclose(x, a @ w.directions)
    nz = a[a > 0]
    assert nz.min() >= 0.5 and nz.max() <= 1.5


def test_expected_l0_matches_empirical_and_zipf_keeps_it():
    for zipf in (0.0, 1.0):
        w = make_world(50, 20, l0=4.0, zipf=zipf, seed=0)
        assert w.expected_l0 == pytest.approx(4.0, abs=1e-6)
        _, a = w.sample(20000, np.random.default_rng(1))
        assert ((a > 0).sum(axis=1).mean()) == pytest.approx(4.0, abs=0.1)
    assert make_world(50, 20, zipf=1.0).probs[0] > make_world(50, 20, zipf=1.0).probs[-1]


def test_orthogonal_world_has_zero_coherence_and_random_grows_with_ratio():
    assert make_world(8, 8, orthogonal=True).coherence() < 1e-12
    with pytest.raises(ValueError):
        make_world(9, 8, orthogonal=True)
    lo = np.mean([make_world(16, 32, seed=s).coherence() for s in range(5)])
    hi = np.mean([make_world(128, 32, seed=s).coherence() for s in range(5)])
    assert hi > lo


# ---------------------------------------------------------------- gradients
def _fd_check(sae, x, tol=1e-5, eps=1e-6):
    loss, grads, _ = sae.loss_and_grads(x)
    rng = np.random.default_rng(0)
    for name, p in sae.params.items():
        flat = p.reshape(-1)
        for i in rng.choice(flat.size, size=min(12, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + eps
            lp = sae.loss_and_grads(x)[0]
            flat[i] = old - eps
            lm = sae.loss_and_grads(x)[0]
            flat[i] = old
            fd = (lp - lm) / (2 * eps)
            g = grads[name].reshape(-1)[i]
            assert abs(fd - g) <= tol + 1e-4 * abs(fd), (name, i, fd, g)


@pytest.mark.parametrize("mode,kw", [("relu", {"l1": 0.0}), ("relu", {"l1": 0.3}),
                                     ("topk", {"k": 2}), ("batchtopk", {"k": 2})])
def test_gradients_match_finite_differences(mode, kw):
    rng = np.random.default_rng(3)
    sae = SAE(6, 14, mode=mode, seed=1, **kw)
    sae.params["b_enc"] += rng.normal(0, 0.1, 14)   # random biases so b_dec/b_enc grads are non-trivial
    sae.params["b_dec"] += rng.normal(0, 0.1, 6)
    x = rng.normal(size=(32, 6))
    _fd_check(sae, x)


def test_a_wrong_encoder_gradient_would_be_caught():
    sae = SAE(5, 10, mode="relu", l1=0.1, seed=0)
    x = np.random.default_rng(0).normal(size=(16, 5))
    _, grads, _ = sae.loss_and_grads(x)
    p, eps = sae.params["W_enc"], 1e-6
    errs = []
    for scale in (1.0, 1.5):
        worst = 0.0
        for i, j in [(0, 0), (1, 3), (2, 5), (3, 7), (4, 9)]:
            old = p[i, j]
            p[i, j] = old + eps
            lp = sae.loss_and_grads(x)[0]
            p[i, j] = old - eps
            lm = sae.loss_and_grads(x)[0]
            p[i, j] = old
            worst = max(worst, abs((lp - lm) / (2 * eps) - scale * grads["W_enc"][i, j]))
        errs.append(worst)
    assert errs[0] < 1e-6 and errs[1] > 1e-3


# ---------------------------------------------------------------- SAE structure
def test_topk_has_at_most_k_active_per_row():
    sae = SAE(8, 32, mode="topk", k=3, seed=0)
    x = np.random.default_rng(0).normal(size=(200, 8))
    a = sae.encode(x)
    assert ((a > 0).sum(axis=1) <= 3).all()
    assert ((a > 0).sum(axis=1) >= 1).any()


def test_batchtopk_uses_k_per_sample_on_average_during_training_and_threshold_at_inference():
    sae = SAE(8, 64, mode="batchtopk", k=4, seed=0)
    x = np.random.default_rng(0).normal(size=(256, 8))
    _, _, info = sae.loss_and_grads(x)
    assert info["active"] == pytest.approx(4.0, abs=0.05)
    assert sae.threshold > 0
    per_row = (info["a"] > 0).sum(axis=1)
    assert per_row.max() > per_row.min()            # rows differ: that is the point
    # inference keeps exactly the activations at or above the tracked threshold
    a = sae.encode(x)
    pre = np.maximum((x - sae.params["b_dec"]) @ sae.params["W_enc"] + sae.params["b_enc"], 0)
    assert np.array_equal(a > 0, pre >= sae.threshold)


def test_normalize_decoder_preserves_function_and_makes_rows_unit():
    rng = np.random.default_rng(0)
    sae = SAE(6, 12, mode="relu", seed=0)
    sae.params["W_dec"] *= rng.uniform(0.5, 3.0, size=(12, 1))
    sae.params["b_enc"] += rng.normal(0, 0.2, 12)
    x = rng.normal(size=(40, 6))
    # scaling a decoder row by c is only function-preserving if the encoder column and bias scale by 1/c
    nrm = np.linalg.norm(sae.params["W_dec"], axis=1)
    sae.params["W_enc"] /= nrm[None, :]
    sae.params["b_enc"] /= nrm
    scaled = sae.reconstruct(x)
    sae.normalize_decoder()
    after = sae.reconstruct(x)
    assert np.allclose(np.linalg.norm(sae.params["W_dec"], axis=1), 1.0)
    assert np.allclose(scaled, after)


def test_decoder_gradient_projection_is_orthogonal_to_rows():
    sae = SAE(6, 12, mode="relu", l1=0.1, seed=0)
    x = np.random.default_rng(0).normal(size=(30, 6))
    _, g, _ = sae.loss_and_grads(x)
    sae.project_decoder_grad(g)
    assert np.abs((g["W_dec"] * sae.params["W_dec"]).sum(axis=1)).max() < 1e-10


def test_invalid_configs_raise():
    with pytest.raises(ValueError):
        SAE(4, 8, mode="nope")
    with pytest.raises(ValueError):
        SAE(4, 8, mode="topk")
    with pytest.raises(ValueError):
        SAE(4, 8, mode="batchtopk", k=9)


# ---------------------------------------------------------------- metrics
def test_recovery_perfect_and_random_and_one_to_one():
    w = make_world(20, 16, seed=0)
    r = feature_recovery(w.directions, w.directions)
    assert r["recovered"] == 1.0 and r["matched"] == 1.0 and r["mean_best_cos"] == pytest.approx(1.0)
    rnd = np.random.default_rng(9).normal(size=(20, 16))
    assert feature_recovery(w.directions, rnd)["recovered"] < 0.2
    # a dictionary that repeats one row 20 times "recovers" 1 feature only once under matching
    dup = np.repeat(w.directions[:1], 20, axis=0)
    r = feature_recovery(w.directions, dup)
    assert r["matched"] == pytest.approx(1 / 20)


def test_purity_ignores_dead_latents_when_given_alive_mask():
    w = make_world(8, 16, seed=0)
    dec = np.vstack([w.directions, np.random.default_rng(99).normal(size=(8, 16))])
    assert latent_purity(w.directions, dec) == pytest.approx(0.5)
    alive = np.array([True] * 8 + [False] * 8)
    assert latent_purity(w.directions, dec, alive=alive) == 1.0
    assert latent_purity(w.directions, dec, alive=np.zeros(16, bool)) == 0.0


def test_fvu_l0_dead_and_ratio_and_recall():
    x = np.random.default_rng(0).normal(size=(100, 5))
    assert fvu(x, x) == 0.0
    assert fvu(x, np.tile(x.mean(axis=0), (100, 1))) == pytest.approx(1.0)
    a = np.array([[1.0, 0, 0], [0, 2.0, 0], [3.0, 1.0, 0]])
    assert l0(a) == pytest.approx(4 / 3)
    assert dead_latents(a).tolist() == [False, False, True]
    w = make_world(6, 12, seed=0)
    true_a = np.abs(np.random.default_rng(0).normal(size=(50, 6))) + 0.1
    half = true_a * 0.5
    assert coefficient_ratio(true_a, half, w.directions, w.directions) == pytest.approx(0.5)
    silent = half.copy()
    silent[:25, 0] = 0.0
    assert feature_recall(true_a, silent, w.directions, w.directions) == pytest.approx(1 - 25 / 300)


# ---------------------------------------------------------------- training
def test_training_recovers_an_orthogonal_world_with_every_mode():
    w = make_world(8, 8, l0=1.5, orthogonal=True, seed=0)
    x, a = w.sample(2000, np.random.default_rng(7))
    for mode, kw in [("topk", {"k": 2}), ("batchtopk", {"k": 2}), ("relu", {"l1": 0.2})]:
        sae = SAE(8, 16, mode=mode, seed=0, **kw)
        fit(sae, w, steps=1500, batch=256)
        r = feature_recovery(w.directions, sae.params["W_dec"], thresh=0.95)
        assert r["matched"] >= 0.99, (mode, r)
        assert fvu(x, sae.reconstruct(x)) < 0.2, mode


def test_training_is_deterministic():
    w = make_world(16, 8, seed=0)
    outs = []
    for _ in range(2):
        sae = SAE(8, 16, mode="topk", k=2, seed=3)
        fit(sae, w, steps=60, batch=64, seed=5)
        outs.append(sae.params["W_dec"].copy())
    assert np.array_equal(outs[0], outs[1])


def test_decoder_stays_unit_norm_during_training():
    w = make_world(16, 8, seed=0)
    sae = SAE(8, 24, mode="relu", l1=0.1, seed=0)
    fit(sae, w, steps=50, batch=64)
    assert np.allclose(np.linalg.norm(sae.params["W_dec"], axis=1), 1.0)


def test_resample_dead_resets_only_dead_latents_toward_residuals():
    w = make_world(16, 8, seed=0)
    sae = SAE(8, 24, mode="topk", k=2, seed=0)
    x, _ = w.sample(512, np.random.default_rng(0))
    dead = np.zeros(24, bool)
    dead[[3, 10, 20]] = True
    before = {k: v.copy() for k, v in sae.params.items()}
    n = resample_dead(sae, x, dead, np.random.default_rng(1))
    assert n == 3
    live = ~dead
    assert np.array_equal(sae.params["W_dec"][live], before["W_dec"][live])
    assert not np.array_equal(sae.params["W_dec"][dead], before["W_dec"][dead])
    assert np.allclose(np.linalg.norm(sae.params["W_dec"][dead], axis=1), 1.0)
    assert resample_dead(sae, x, np.zeros(24, bool), np.random.default_rng(1)) == 0
