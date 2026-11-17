import numpy as np
import pytest
from bayesopt import ParEGO, pareto_mask, hypervolume_2d
from bayesopt.multiobj import chebyshev


def test_pareto_mask_by_hand():
    Y = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [2, 2]])
    assert pareto_mask(Y).tolist() == [True, True, True, False, True]


def test_hypervolume_by_hand():
    assert hypervolume_2d([[1, 1]], [3, 3]) == 4.0
    # staircase: (1,2) and (2,1), ref (3,3): 2*1 + 1*1 + ... = area 3
    assert hypervolume_2d([[1, 2], [2, 1]], [3, 3]) == 3.0
    assert hypervolume_2d([[4, 4]], [3, 3]) == 0.0
    assert hypervolume_2d([[1, 2], [2, 1], [2.5, 2.5]], [3, 3]) == 3.0  # dominated point adds nothing


def test_hypervolume_matches_monte_carlo():
    rng = np.random.default_rng(1)
    Y = rng.random((12, 2)); ref = np.array([1.2, 1.2])
    U = rng.random((200000, 2)) * ref
    dom = ((Y[None] <= U[:, None]).all(2)).any(1)
    assert abs(hypervolume_2d(Y, ref) - dom.mean() * ref.prod()) < 0.01


def test_chebyshev_prefers_balanced_point():
    Yn = np.array([[0.0, 1.0], [0.5, 0.5]])
    s = chebyshev(Yn, np.array([0.5, 0.5]))
    assert s[1] < s[0]


def test_tell_validates():
    with pytest.raises(ValueError):
        ParEGO([[0, 1]], n_obj=2).tell([0.5], [1.0])


def schaffer(x):  # convex front: f1=x^2, f2=(x-2)^2 on x in [-2, 4], Pareto set x in [0,2]
    return [x[0] ** 2, (x[0] - 2) ** 2]


def test_parego_beats_random_on_schaffer():
    ref = [20.0, 20.0]
    hv_p, hv_r = [], []
    for s in range(3):
        p = ParEGO([[-2, 4]], rng=np.random.default_rng(s))
        _, Y = p.run(schaffer, 20)
        hv_p.append(hypervolume_2d(Y, ref))
        r = np.random.default_rng(100 + s).uniform(-2, 4, 20)
        hv_r.append(hypervolume_2d(np.array([schaffer([v]) for v in r]), ref))
    assert np.mean(hv_p) > np.mean(hv_r)
    X, Y = p.front()
    assert len(X) >= 3 and (X >= -0.2).all() and (X <= 2.2).all()  # front lies near the true Pareto set [0, 2]
