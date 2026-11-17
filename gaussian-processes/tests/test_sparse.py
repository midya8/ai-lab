import numpy as np
import pytest

from gaussproc import RBF, ExactGP, SparseGP, greedy_inducing, kmeans_inducing, sample_gp_data


@pytest.fixture
def problem():
    X, y, _ = sample_gp_data(RBF(0.9, 1.2), 60, 0.1, seed=4)
    return X, y, RBF(0.9, 1.2)


@pytest.mark.parametrize("method", ["vfe", "fitc", "dtc"])
def test_with_Z_equal_X_every_method_recovers_the_exact_gp(problem, method):
    X, y, k = problem
    ex = ExactGP(k.copy(), 0.1).set_data(X, y)
    sp = SparseGP(k.copy(), X, 0.1, method).set_data(X, y)
    assert abs(sp.bound() - ex.log_marginal_likelihood()) < 1e-3
    Xs = np.linspace(-3, 3, 15)[:, None]
    me, ve = ex.predict(Xs)
    ms, vs = sp.predict(Xs)
    assert np.allclose(ms, me, atol=1e-4) and np.allclose(vs, ve, atol=1e-4)


def test_vfe_is_a_lower_bound_on_the_exact_lml(problem):
    X, y, k = problem
    exact = ExactGP(k.copy(), 0.1).set_data(X, y).log_marginal_likelihood()
    for seed in range(5):
        Z = X[np.random.default_rng(seed).choice(60, 8, replace=False)]
        assert SparseGP(k.copy(), Z, 0.1, "vfe").set_data(X, y).bound() <= exact + 1e-9


def test_vfe_bound_increases_with_nested_inducing_sets(problem):
    X, y, k = problem
    idx = greedy_inducing(k, X, 30)
    bounds = [SparseGP(k.copy(), X[idx[:m]], 0.1, "vfe").set_data(X, y).bound() for m in (2, 4, 8, 16, 30)]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(bounds, bounds[1:]))


def test_collapsed_vfe_matches_dense_formula(problem):
    X, y, k = problem
    Z = X[:7]
    s2 = 0.1
    Kuu = k(Z) + 1e-8 * np.eye(7)
    Kuf = k(Z, X)
    Q = Kuf.T @ np.linalg.solve(Kuu, Kuf)
    C = Q + s2 * np.eye(60)
    ref = (-0.5 * y @ np.linalg.solve(C, y) - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * 60 * np.log(2 * np.pi)
           - np.trace(k(X) - Q) / (2 * s2))
    assert np.isclose(SparseGP(k.copy(), Z, s2, "vfe").set_data(X, y).bound(), ref, atol=1e-5)


def test_fitc_uses_heteroscedastic_noise_and_dtc_does_not(problem):
    X, y, k = problem
    Z = X[:5]
    f = SparseGP(k.copy(), Z, 0.1, "fitc").set_data(X, y)
    d = SparseGP(k.copy(), Z, 0.1, "dtc").set_data(X, y)
    assert np.allclose(d._lam, 0.1)
    assert np.allclose(f._lam, 0.1 + f._resid) and f._lam.std() > 0
    assert f.bound() != d.bound()


def test_sparse_predictions_converge_to_the_exact_gp_as_M_grows(problem):
    X, y, k = problem
    ex = ExactGP(k.copy(), 0.1).set_data(X, y)
    Xs = np.linspace(-3, 3, 40)[:, None]
    me, ve = ex.predict(Xs)
    idx = greedy_inducing(k, X, 40)
    gaps = []
    for m in (2, 4, 8, 40):
        ms, vs = SparseGP(k.copy(), X[idx[:m]], 0.1, "vfe").set_data(X, y).predict(Xs)
        gaps.append(np.abs(ms - me).max() + np.abs(vs - ve).max())
    assert gaps[0] > gaps[1] > gaps[2] > gaps[3] and gaps[3] < 1e-3


def test_greedy_selection_is_distinct_and_shrinks_the_trace(problem):
    X, y, k = problem
    idx = greedy_inducing(k, X, 12)
    assert len(set(idx.tolist())) == 12
    gaps = [SparseGP(k.copy(), X[idx[:m]], 0.1).set_data(X, y).trace_gap() for m in (1, 4, 12)]
    assert gaps[0] > gaps[1] > gaps[2]
    rnd = np.mean([SparseGP(k.copy(), X[np.random.default_rng(s).choice(60, 12, replace=False)], 0.1)
                   .set_data(X, y).trace_gap() for s in range(10)])
    assert gaps[2] < rnd


def test_greedy_pick_is_the_largest_remaining_variance(problem):
    X, y, k = problem
    K = k(X)
    idx = greedy_inducing(k, X, 6)
    # the last pick had the largest Schur-complement variance given the first five
    last_before = np.diag(K) - np.sum(K[:, idx[:5]] * np.linalg.solve(K[np.ix_(idx[:5], idx[:5])] + 1e-10 * np.eye(5), K[idx[:5]]).T, axis=1)
    assert np.isclose(last_before.max(), last_before[idx[5]], atol=1e-6)


def test_kmeans_centres_cover_the_data():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(-3, 0.1, (50, 1)), rng.normal(3, 0.1, (50, 1))])
    C = np.sort(kmeans_inducing(X, 2, seed=0)[:, 0])
    assert abs(C[0] + 3) < 0.2 and abs(C[1] - 3) < 0.2


def test_fit_raises_objective_and_fitc_underestimates_noise_vs_vfe():
    X, y, _ = sample_gp_data(RBF(0.8, 1.0), 300, 0.2, seed=1, low=-4, high=4)
    k = RBF(1.0, 1.0)
    Z = X[greedy_inducing(k, X, 8)]
    res = {}
    for m in ("vfe", "fitc"):
        sp = SparseGP(RBF(1.0, 1.0), Z, 0.5, m).set_data(X, y)
        before = sp.bound()
        sp.fit(X, y)
        assert sp.bound() > before
        res[m] = sp.noise
    assert res["fitc"] < res["vfe"]          # the known FITC tendency


def test_bad_method_rejected():
    with pytest.raises(ValueError):
        SparseGP(RBF(), np.zeros((2, 1)), 0.1, "nope")
