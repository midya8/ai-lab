import numpy as np
import pytest
from bayesopt import BayesOpt, BENCHMARKS, halton, random_search, branin, hartmann3, noisy, optimize_acquisition


def test_benchmark_optima_are_consistent():
    for b in BENCHMARKS.values():
        for x in b.argmins:
            assert b(x) == pytest.approx(b.optimum, abs=2e-4), b.name
        lo, hi = b.bounds[:, 0], b.bounds[:, 1]
        assert ((b.argmins >= lo - 1e-9) & (b.argmins <= hi + 1e-9)).all()


def test_benchmark_values_are_above_optimum_on_random_points():
    rng = np.random.default_rng(0)
    for b in BENCHMARKS.values():
        X = b.bounds[:, 0] + rng.random((300, b.dim)) * (b.bounds[:, 1] - b.bounds[:, 0])
        assert min(b(x) for x in X) >= b.optimum - 1e-6, b.name


def test_halton_is_in_unit_cube_distinct_and_low_discrepancy():
    H = halton(256, 3)
    assert H.min() >= 0 and H.max() < 1 and len({tuple(r) for r in H}) == 256
    for j in range(3):                                   # every dyadic-ish 1/8 bin is near-uniformly filled
        counts = np.histogram(H[:, j], bins=8, range=(0, 1))[0]
        assert counts.min() >= 256 / 8 - 6 and counts.max() <= 256 / 8 + 6


def test_unit_cube_roundtrip():
    bo = BayesOpt(branin.bounds)
    X = np.array([[-5.0, 0.0], [10.0, 15.0], [2.5, 7.5]])
    assert np.allclose(bo.from_unit(bo.to_unit(X)), X)
    assert np.allclose(bo.to_unit(X)[2], [0.5, 0.5])


def test_optimize_acquisition_finds_peak_of_smooth_score():
    peak = np.array([0.31, 0.77, 0.12])
    score = lambda U: -((U - peak) ** 2).sum(1)
    u = optimize_acquisition(score, 3, np.random.default_rng(0))
    assert np.abs(u - peak).max() < 0.02


def test_ask_is_deterministic_for_a_seed_and_respects_bounds():
    def history(seed):
        bo = BayesOpt(branin.bounds, rng=np.random.default_rng(seed))
        bo.run(branin, 12)
        return bo.X, bo.y
    X1, y1 = history(4); X2, y2 = history(4)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)
    assert (X1 >= branin.bounds[:, 0] - 1e-9).all() and (X1 <= branin.bounds[:, 1] + 1e-9).all()


@pytest.mark.parametrize("acq", ["ei", "ucb", "ts"])
def test_bo_beats_random_search_on_branin(acq):
    n, seeds = 25, range(4)
    bo_regret, rs_regret = [], []
    for s in seeds:
        bo = BayesOpt(branin.bounds, acq, rng=np.random.default_rng(s))
        bo_regret.append(bo.run(branin, n)[-1] - branin.optimum)
        _, y = random_search(branin, branin.bounds, n, np.random.default_rng(s))
        rs_regret.append(y.min() - branin.optimum)
    assert np.median(bo_regret) < 0.2 * np.median(rs_regret)
    assert np.median(bo_regret) < 0.1


def test_best_so_far_is_monotone_and_never_below_optimum():
    bo = BayesOpt(hartmann3.bounds, rng=np.random.default_rng(0))
    h = bo.run(hartmann3, 20)
    assert (np.diff(h) <= 0).all() and h[-1] >= hartmann3.optimum - 1e-9
    x, y = bo.best()
    assert y == h[-1] and hartmann3(x) == pytest.approx(y)


def test_batch_ask_returns_distinct_points_and_runs():
    bo = BayesOpt(branin.bounds, rng=np.random.default_rng(1))
    bo.run(branin, 10)
    X = bo.ask(q=4)
    assert X.shape == (4, 2) and len({tuple(np.round(r, 6)) for r in X}) == 4
    bo2 = BayesOpt(branin.bounds, rng=np.random.default_rng(1))
    h = bo2.run(branin, 22, q=4)
    assert len(bo2.y) == 22 and h[-1] < 1.5


def test_ts_batch_gives_distinct_points():
    bo = BayesOpt(branin.bounds, "ts", rng=np.random.default_rng(2))
    bo.run(branin, 10)
    X = bo.ask(q=3)
    assert len({tuple(np.round(r, 6)) for r in X}) == 3


def test_initial_design_is_served_before_model_is_fit():
    bo = BayesOpt(branin.bounds, n_init=6)
    X = bo.ask(q=10)
    assert len(X) == 6 and bo.gp is None


def test_noisy_objective_incumbent_uses_posterior_mean():
    rng = np.random.default_rng(0)
    nb = noisy(branin, 3.0, rng)
    bo = BayesOpt(nb.bounds, noisy=True, rng=np.random.default_rng(0))
    bo.run(nb, 30)
    gp = bo._fit()
    inc = bo.incumbent(gp)
    assert inc > bo.y.min()                 # smoothed incumbent is not the luckiest noisy draw


def test_bad_acquisition_rejected():
    with pytest.raises(ValueError):
        BayesOpt(branin.bounds, "magic")


def test_quasi_random_baseline_differs_across_seeds_and_stays_in_bounds():
    X1, _ = random_search(branin, branin.bounds, 16, np.random.default_rng(0), quasi=True)
    X2, _ = random_search(branin, branin.bounds, 16, np.random.default_rng(1), quasi=True)
    assert not np.allclose(X1, X2)
    assert (X1 >= branin.bounds[:, 0]).all() and (X1 <= branin.bounds[:, 1]).all()


def test_gradient_refinement_never_lowers_ei_and_stays_in_bounds():
    from bayesopt import fit_gp, expected_improvement, refine_ei_gradient
    rng = np.random.default_rng(3)
    U = rng.random((12, 2))
    y = np.array([branin(branin.bounds[:, 0] + u * (branin.bounds[:, 1] - branin.bounds[:, 0])) for u in U])
    gp = fit_gp(U, y, rng=rng)
    best = y.min()
    gains = 0
    for _ in range(10):
        x0 = rng.random(2)
        x1 = refine_ei_gradient(gp, x0, best)
        e0 = expected_improvement(*gp.predict(x0[None]), best)[0]
        e1 = expected_improvement(*gp.predict(x1[None]), best)[0]
        assert (x1 >= 0).all() and (x1 <= 1).all()
        assert e1 >= e0 - 1e-12
        gains += e1 > e0 + 1e-9
    assert gains >= 5


def test_bayesopt_grad_refine_runs_and_improves_on_init():
    bo = BayesOpt(branin.bounds, acquisition="ei", grad_refine=True, rng=np.random.default_rng(1))
    bo.run(branin, 20)
    assert len(bo.y) == 20 and bo.best()[1] < 5.0 and bo.best()[1] <= bo.y[:bo.n_init].min()
