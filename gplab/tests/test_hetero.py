import numpy as np

from gplab import RBF, ExactGP, HeteroGP, coverage, gaussian_nlpd


def make(seed, n=200, nt=400):
    rng = np.random.default_rng(seed)
    sd = lambda x: 0.05 + 0.4 * (x[:, 0] + 3) / 6
    X = np.sort(rng.uniform(-3, 3, (n, 1)), axis=0)
    y = np.sin(X[:, 0]) + sd(X) * rng.standard_normal(n)
    Xt = rng.uniform(-3, 3, (nt, 1))
    yt = np.sin(Xt[:, 0]) + sd(Xt) * rng.standard_normal(nt)
    return X, y, Xt, yt, sd


def test_noise_function_recovered():
    cors = []
    for seed in range(3):
        X, y, _, _, sd = make(seed)
        h = HeteroGP(RBF(1.5, 1.0), RBF(1.5, 1.0)).fit(X, y)
        g = np.linspace(-2.5, 2.5, 50)[:, None]
        cors.append(np.corrcoef(np.log(h.noise_at(g)), np.log(sd(g) ** 2))[0, 1])
    assert min(cors) > 0.8


def test_beats_homoscedastic_nlpd():
    for seed in range(3):
        X, y, Xt, yt, _ = make(seed)
        h = HeteroGP(RBF(1.5, 1.0), RBF(1.5, 1.0)).fit(X, y)
        m, v = h.predict(Xt, include_noise=True)
        gp = ExactGP(RBF(1.5, 1.0), 0.1).fit(X, y)
        m0, v0 = gp.predict(Xt, include_noise=True)
        assert gaussian_nlpd(yt, m, v) < gaussian_nlpd(yt, m0, v0)
        assert 0.88 < coverage(yt, m, v) < 0.99


def test_constant_noise_gives_flat_estimate():
    rng = np.random.default_rng(0)
    X = np.sort(rng.uniform(-3, 3, (150, 1)), axis=0)
    y = np.sin(X[:, 0]) + 0.2 * rng.standard_normal(150)
    h = HeteroGP(RBF(1.5, 1.0), RBF(2.0, 0.5)).fit(X, y)
    ratio = h.noise_.max() / h.noise_.min()
    assert ratio < 6 and abs(np.median(h.noise_) - 0.04) < 0.03
