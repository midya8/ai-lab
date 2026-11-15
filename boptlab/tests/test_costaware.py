import numpy as np
import pytest
from boptlab import CostAwareBayesOpt, best_at_budget


def f(x):
    return float((x[0] - 0.3) ** 2 + (x[1] - 0.6) ** 2)


def cost(x):  # expensive where x0 is large
    return float(np.exp(3 * x[0]))


def test_budget_respected_and_trace_monotone():
    bo = CostAwareBayesOpt([[0, 1], [0, 1]], budget=30, rng=np.random.default_rng(0))
    spent, best = bo.run(f, cost)
    assert spent[-1] <= 30 and bo.spent <= 30
    assert (np.diff(best) <= 1e-12).all() and (np.diff(spent) > 0).all()


def test_tell_validates_cost():
    bo = CostAwareBayesOpt([[0, 1]], budget=1)
    with pytest.raises(ValueError):
        bo.tell([0.5], 1.0, 0.0)


def test_best_at_budget():
    tr = (np.array([1., 3., 6.]), np.array([5., 2., 1.]))
    assert best_at_budget(tr, 0.5) == float("inf")
    assert best_at_budget(tr, 3) == 2.0 and best_at_budget(tr, 100) == 1.0


def test_cost_aware_makes_more_evals_than_blind():
    n = []
    for aware in (True, False):
        tot = 0
        for s in range(3):
            bo = CostAwareBayesOpt([[0, 1], [0, 1]], budget=25, rng=np.random.default_rng(s), cost_aware=aware)
            bo.run(f, cost); tot += len(bo.y)
        n.append(tot)
    assert n[0] >= n[1]
