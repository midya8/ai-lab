import numpy as np
from boptlab import BayesOpt, branin, max_value_entropy_search


def test_mes_nonnegative_and_finite():
    mu = np.linspace(-2, 3, 50); sd = np.linspace(0.05, 2, 50)
    a = max_value_entropy_search(mu, sd, np.array([-1.0, -0.5, -1.5]))
    assert np.isfinite(a).all() and (a >= -1e-12).all()


def test_mes_prefers_uncertain_point_near_optimum():
    ys = np.array([0.0, -0.1])
    certain_bad = max_value_entropy_search([2.0], [0.01], ys)[0]
    uncertain = max_value_entropy_search([0.5], [1.0], ys)[0]
    assert uncertain > certain_bad


def test_mes_optimizer_beats_random_on_branin():
    bo = BayesOpt(branin.bounds, acquisition="mes", rng=np.random.default_rng(1))
    hist = bo.run(branin, 25)
    assert hist[-1] < 1.5 and hist[-1] >= branin.optimum - 1e-6
