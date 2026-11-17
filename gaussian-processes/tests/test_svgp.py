import numpy as np
import pytest

from gaussproc import RBF, SVGP, SparseGP, sample_gp_data, kmeans_inducing


@pytest.fixture
def problem():
    X, y, _ = sample_gp_data(RBF(0.9, 1.2), 200, 0.1, seed=3)
    Z = kmeans_inducing(X, 12)
    return X, y, Z


def test_full_batch_rho1_equals_collapsed_vfe(problem):
    X, y, Z = problem
    k = RBF(0.9, 1.2)
    vfe = SparseGP(k.copy(), Z, 0.1, "vfe", center=True).set_data(X, y)
    sv = SVGP(k.copy(), Z, 0.1)
    sv.mean_ = float(y.mean())
    sv.step(X, y, X.shape[0], 1.0)
    Xs = np.linspace(-3, 3, 25)[:, None]
    m1, v1 = vfe.predict(Xs)
    m2, v2 = sv.predict(Xs)
    assert np.allclose(m1, m2, atol=1e-6) and np.allclose(v1, v2, atol=1e-6)


def test_optimal_q_maximises_elbo_and_matches_vfe_bound(problem):
    X, y, Z = problem
    k = RBF(0.9, 1.2)
    sv = SVGP(k.copy(), Z, 0.1)
    sv.mean_ = float(y.mean())
    sv.step(X, y, X.shape[0], 1.0)
    best = sv.elbo(X, y)
    vfe = SparseGP(k.copy(), Z, 0.1, "vfe", center=True).set_data(X, y).bound()
    assert abs(best - vfe) < 1e-6
    sv.h = sv.h + 0.3          # any perturbation lowers the ELBO
    assert sv.elbo(X, y) < best


def test_minibatch_training_approaches_collapsed_predictor(problem):
    X, y, Z = problem
    k = RBF(0.9, 1.2)
    sv = SVGP(k.copy(), Z, 0.1, seed=1).fit(X, y, epochs=60, batch=40)
    vfe = SparseGP(k.copy(), Z, 0.1, "vfe", center=True).set_data(X, y)
    Xs = np.linspace(-3, 3, 40)[:, None]
    assert np.max(np.abs(sv.predict(Xs)[0] - vfe.predict(Xs)[0])) < 0.05
    assert sv.elbo(X, y) > vfe.bound() - 3.0


def test_elbo_improves_with_training(problem):
    X, y, Z = problem
    sv = SVGP(RBF(0.9, 1.2), Z, 0.1, seed=0)
    sv.mean_ = float(y.mean())
    before = sv.elbo(X, y)
    sv.fit(X, y, epochs=5, batch=50)
    assert sv.elbo(X, y) > before


def test_bad_rho_rejected(problem):
    X, y, Z = problem
    sv = SVGP(RBF(), Z)
    sv.mean_ = 0.0
    with pytest.raises(ValueError):
        sv.step(X[:5], y[:5], 5, 0.0)
