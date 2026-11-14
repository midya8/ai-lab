import numpy as np
import pytest
from boptlab import ConstrainedBayesOpt, prob_feasible, random_search


def f(x):  # unconstrained optimum (0.9, 0.9) is infeasible; constrained optimum lies on the disc boundary
    return float((x[0] - 0.9) ** 2 + (x[1] - 0.9) ** 2)


def cons(x):  # feasible inside the disc of radius 0.5 around (0.3, 0.3)
    return [float((x[0] - 0.3) ** 2 + (x[1] - 0.3) ** 2 - 0.25)]


def test_prob_feasible_values():
    assert prob_feasible(0.0, 1.0) == pytest.approx(0.5)
    assert prob_feasible(-10.0, 1.0) > 0.999999
    assert prob_feasible(1.0, 0.0) == 0.0 and prob_feasible(-1.0, 0.0) == 1.0


def test_tell_and_feasible_bookkeeping():
    bo = ConstrainedBayesOpt([[0, 1], [0, 1]], 1)
    bo.tell([[0.3, 0.3], [0.9, 0.9]], [f([.3, .3]), f([.9, .9])], [cons([.3, .3]), cons([.9, .9])])
    assert bo.feasible_mask().tolist() == [True, False]
    assert bo.best()[1] == pytest.approx(f([.3, .3]))
    assert bo.trace().tolist() == pytest.approx([f([.3, .3])] * 2)
    bo2 = ConstrainedBayesOpt([[0, 1], [0, 1]], 1)
    bo2.tell([[0.9, 0.9]], [f([.9, .9])], [cons([.9, .9])])
    assert np.isnan(bo2.trace()).all() and bo2.best() is None


def test_finds_boundary_optimum_and_respects_constraint():
    bo = ConstrainedBayesOpt([[0, 1], [0, 1]], 1, rng=np.random.default_rng(1))
    tr = bo.run(f, cons, 28)
    x, v = bo.best()
    assert cons(x)[0] <= 0
    opt = (0.6 * np.sqrt(2) - 0.5) ** 2  # (distance from (.9,.9) to disc centre) - radius, squared
    assert v < opt + 0.02
    assert (np.diff(tr[~np.isnan(tr)]) <= 1e-12).all()


def test_beats_random_search_on_feasible_value():
    wins = 0
    for s in range(4):
        bo = ConstrainedBayesOpt([[0, 1], [0, 1]], 1, rng=np.random.default_rng(s))
        bo.run(f, cons, 24)
        X, y = random_search(f, [[0, 1], [0, 1]], 24, np.random.default_rng(100 + s))
        ok = np.array([cons(x)[0] <= 0 for x in X])
        rs = y[ok].min() if ok.any() else np.inf
        wins += bo.best()[1] <= rs
    assert wins >= 3
