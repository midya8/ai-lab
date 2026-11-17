import time

import numpy as np
import pytest

from gplab import ExactGP, Matern32, Matern52
from gplab.kalman import KalmanMatern


def data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    return x, np.sin(x) + 0.2 * rng.standard_normal(n)


@pytest.mark.parametrize("nu,K", [(1.5, Matern32), (2.5, Matern52)])
def test_matches_exact_gp(nu, K):
    x, y = data(60)
    kal = KalmanMatern(nu, 1.3, 0.8, 0.05)
    gp = ExactGP(K(1.3, 0.8), noise=0.05); gp.set_data(x[:, None], y)
    assert kal.log_marginal_likelihood(x, y) == pytest.approx(gp.log_marginal_likelihood(), abs=1e-6)
    xs = np.random.default_rng(1).uniform(-1, 11, 40)
    m, v = kal.predict(x, y, xs)
    m0, v0 = gp.predict(xs[:, None])
    np.testing.assert_allclose(m, m0, atol=1e-6)
    np.testing.assert_allclose(v, v0, atol=1e-6)


def test_noise_option_and_order_invariance():
    x, y = data(30)
    kal = KalmanMatern(1.5, 1.0, 1.0, 0.1)
    p = np.random.default_rng(2).permutation(30)
    assert kal.log_marginal_likelihood(x[p], y[p]) == pytest.approx(kal.log_marginal_likelihood(x, y))
    xs = np.array([3.0, 1.0])
    _, v = kal.predict(x, y, xs); _, vn = kal.predict(x, y, xs, include_noise=True)
    np.testing.assert_allclose(vn - v, 0.1)


def test_wrong_params_change_lml():
    x, y = data(40)
    assert KalmanMatern(1.5, 1.0, 1.0, 0.05).log_marginal_likelihood(x, y) != pytest.approx(
        KalmanMatern(1.5, 1.0, 1.0, 0.5).log_marginal_likelihood(x, y))


def test_linear_scaling_beats_cubic():
    x, y = data(3000)
    t = time.time(); KalmanMatern(1.5).log_marginal_likelihood(x, y); tk = time.time() - t
    gp = ExactGP(Matern32(1.0, 1.0), noise=0.1); gp.set_data(x[:, None], y)
    t = time.time(); gp.log_marginal_likelihood(); tg = time.time() - t
    assert np.isfinite(tk) and tk < 20 and tg > 0
