import numpy as np
import pytest

from gaussproc import (RBF, ExactGP, FeatureGP, Linear, Matern32, Matern52, NystromFeatures,
                   RandomFourierFeatures, greedy_inducing, kernel_approx_error, sample_gp_data)


def test_rff_kernel_error_shrinks_like_inverse_sqrt_D():
    X = np.random.default_rng(0).uniform(-2, 2, (60, 2))
    k = RBF(1.0, 1.5)
    errs = []
    for D in (50, 200, 800, 3200):
        errs.append(np.mean([kernel_approx_error(k(X), RandomFourierFeatures(k, D, 2, seed=s)(X)) for s in range(8)]))
    assert errs[0] > errs[1] > errs[2] > errs[3]
    assert 0.35 < errs[0] / errs[3] / 8 < 1.6      # 64x features -> ~8x lower error


def test_rff_is_unbiased_for_the_kernel():
    X = np.array([[0.0, 0.0], [0.5, -0.3]])
    k = RBF(0.8, 2.0)
    est = np.mean([RandomFourierFeatures(k, 400, 2, seed=s)(X) @ RandomFourierFeatures(k, 400, 2, seed=s)(X).T
                   for s in range(200)], axis=0)
    assert np.allclose(est, k(X), atol=0.03)


@pytest.mark.parametrize("cls", [Matern32, Matern52])
def test_matern_spectral_sampling_converges(cls):
    X = np.random.default_rng(1).uniform(-2, 2, (40, 1))
    k = cls(0.9, 1.3)
    e = kernel_approx_error(k(X), RandomFourierFeatures(k, 20000, 1, seed=0)(X))
    assert e < 0.12


def test_wrong_spectral_density_is_detected():
    """An RBF (Gaussian) spectrum does not approximate a Matern-3/2 kernel."""
    X = np.random.default_rng(1).uniform(-2, 2, (40, 1))
    k = Matern32(0.9, 1.3)
    wrong = kernel_approx_error(k(X), RandomFourierFeatures(RBF(0.9, 1.3), 20000, 1, seed=0)(X))
    right = kernel_approx_error(k(X), RandomFourierFeatures(k, 20000, 1, seed=0)(X))
    assert wrong > 0.1 and wrong > 5 * right


def test_orthogonal_features_reduce_error():
    X = np.random.default_rng(0).uniform(-2, 2, (60, 4))
    k = RBF(1.5, 1.0)
    iid = np.mean([kernel_approx_error(k(X), RandomFourierFeatures(k, 8, 4, seed=s)(X)) for s in range(60)])
    orf = np.mean([kernel_approx_error(k(X), RandomFourierFeatures(k, 8, 4, seed=s, orthogonal=True)(X)) for s in range(60)])
    assert orf < iid


def test_orthogonal_blocks_are_orthogonal_up_to_row_norms():
    f = RandomFourierFeatures(RBF(1.0, 1.0), 4, 4, seed=0, orthogonal=True)
    W = f.W
    G = W @ W.T
    off = G - np.diag(np.diag(G))
    assert np.abs(off).max() < 1e-9


def test_nystrom_with_Z_equal_X_is_the_exact_kernel_and_exact_gp():
    X, y, _ = sample_gp_data(RBF(1.0, 1.0), 50, 0.1, seed=2)
    k = RBF(1.0, 1.0)
    feat = NystromFeatures(k, X)
    assert np.allclose(feat(X) @ feat(X).T, k(X), atol=1e-6)
    ex = ExactGP(k.copy(), 0.1).set_data(X, y)
    fg = FeatureGP(feat, 0.1).fit(X, y)
    Xs = np.linspace(-3, 3, 20)[:, None]
    assert np.allclose(fg.predict(Xs)[0], ex.predict(Xs)[0], atol=1e-4)
    assert np.allclose(fg.predict(Xs)[1], ex.predict(Xs)[1], atol=1e-4)


def test_nystrom_error_is_the_schur_complement_and_greedy_beats_random():
    X = np.random.default_rng(0).uniform(-3, 3, (200, 1))
    k = RBF(0.7, 1.0)
    errs = {}
    for name in ("greedy", "random"):
        vals = []
        for s in range(5):
            idx = greedy_inducing(k, X, 10, seed=s) if name == "greedy" else np.random.default_rng(s).choice(200, 10, replace=False)
            vals.append(kernel_approx_error(k(X), NystromFeatures(k, X[idx])(X)))
        errs[name] = np.mean(vals)
    assert errs["greedy"] < errs["random"]


def test_feature_gp_converges_to_exact_gp_as_D_grows():
    X, y, _ = sample_gp_data(RBF(1.0, 1.0), 120, 0.1, seed=5, dim=2, low=-2, high=2)
    k = RBF(1.0, 1.0)
    ex = ExactGP(k.copy(), 0.1).set_data(X, y)
    Xs = np.random.default_rng(0).uniform(-2, 2, (80, 2))
    me = ex.predict(Xs)[0]
    gaps = [np.linalg.norm(FeatureGP(RandomFourierFeatures(k, D, 2, seed=0), 0.1).fit(X, y).predict(Xs)[0] - me)
            for D in (20, 200, 2000)]
    assert gaps[0] > gaps[1] > gaps[2] and gaps[2] < 0.15 * np.linalg.norm(me)


def test_feature_gp_variance_is_positive_and_bounded_by_prior():
    X, y, _ = sample_gp_data(RBF(1.0, 1.0), 60, 0.1, seed=6)
    fg = FeatureGP(RandomFourierFeatures(RBF(1.0, 1.0), 100, 1, seed=0), 0.1).fit(X, y)
    _, v = fg.predict(np.linspace(-3, 3, 50)[:, None])
    assert v.min() > 0 and v.max() < 1.6


def test_rff_rejects_unsupported_kernels():
    with pytest.raises(TypeError):
        RandomFourierFeatures(Linear(1.0), 10, 1)
