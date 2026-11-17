import numpy as np
from gaussproc.ard import fit_ard, rank_features, relevance


def _data(n=120, d=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = np.sin(2 * X[:, 0]) + 0.5 * X[:, 2] ** 2 + 0.05 * rng.normal(size=n)
    return X, y


def test_relevance_sums_to_one():
    X, y = _data()
    r = relevance(fit_ard(X, y, restarts=0))
    assert np.isclose(r.sum(), 1) and np.all(r > 0)


def test_informative_dims_ranked_first():
    X, y = _data()
    order = rank_features(X, y)
    assert set(order[:2].tolist()) == {0, 2}


def test_irrelevant_dims_get_long_lengthscales():
    X, y = _data()
    ls = fit_ard(X, y).kernel.lengthscale
    assert min(ls[[1, 3, 4]]) > 2 * max(ls[[0, 2]])


def test_ard_beats_isotropic_on_heldout():
    from gaussproc import ExactGP, RBF
    X, y = _data(160)
    Xt, yt = X[:100], y[:100]; Xv, yv = X[100:], y[100:]
    a = fit_ard(Xt, yt)
    iso = ExactGP(RBF(), noise=0.1, center=True); iso.fit(Xt, yt, restarts=2)
    e = lambda g: np.mean((g.predict(Xv)[0] - yv) ** 2)
    assert e(a) < e(iso)
