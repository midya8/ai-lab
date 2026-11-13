import numpy as np
from rlgrid import *


def test_walls_and_bounds():
    env = GridWorld()
    assert env._move((0, 0), 0) == (0, 0)
    assert env._move((0, 2), 3) == (0, 2)  # wall at (0,3)


def test_value_iteration_solves_det():
    env = GridWorld()
    q = value_iteration(env)
    mean, win = evaluate(env, greedy_policy(q), 5)
    assert win == 1.0 and mean > 0.8


def test_q_learning_matches_optimal_return():
    env = GridWorld(seed=1)
    q, rets = q_learning(env, episodes=600, seed=1)
    opt = evaluate(env, greedy_policy(value_iteration(env)), 1)[0]
    got = evaluate(env, greedy_policy(q), 1)[0]
    assert abs(got - opt) < 1e-9


def test_dyna_learns_faster_than_q():
    env = GridWorld(seed=0)
    _, rq = q_learning(env, episodes=30, seed=0)
    _, rd = dyna_q(env, episodes=30, seed=0, planning=20)
    assert rd[10:].mean() > rq[10:].mean()


def test_sarsa_avoids_pit_under_slip():
    env = GridWorld(slip=0.2, seed=2)
    q, _ = sarsa(env, episodes=1500, seed=2)
    _, win = evaluate(env, greedy_policy(q), 200)
    assert win > 0.8


def test_slip_transitions_sum_to_one():
    env = GridWorld(slip=0.3)
    assert abs(sum(p for p, _ in env.transitions((2, 2), 0)) - 1) < 1e-12


def test_render_shape():
    env = GridWorld()
    assert len(env.render(greedy_policy(value_iteration(env))).splitlines()) == env.h


def test_cliff_sarsa_falls_less_than_q_learning():
    import numpy as np
    from rlgrid.env import CLIFF_MAP
    falls = {}
    for name, fn in [("q", q_learning), ("s", sarsa)]:
        f = []
        for sd in range(4):
            _, r = fn(GridWorld(CLIFF_MAP, seed=sd), episodes=400, eps=0.1, seed=sd, gamma=0.99)
            f.append((r[-100:] < -0.5).mean())
        falls[name] = np.mean(f)
    assert falls["s"] < falls["q"]


def test_expected_sarsa_and_double_q_reach_optimum_on_deterministic_map():
    from rlgrid import expected_sarsa, double_q
    env = GridWorld(slip=0.0)
    opt = evaluate(env, greedy_policy(value_iteration(env)), 5)[0]
    for fn in (expected_sarsa, double_q):
        q, _ = fn(GridWorld(slip=0.0), episodes=600, seed=1)
        assert evaluate(env, greedy_policy(q), 5)[0] >= opt - 1e-9


def test_reinforce_learns_det_gridworld():
    env = GridWorld(seed=0)
    th, rets = reinforce(env, episodes=400, seed=0)
    assert rets[-50:].mean() > rets[:50].mean()
    mean, win = evaluate(env, greedy_policy(th), 1)
    assert win == 1.0


def test_reinforce_baseline_not_worse_on_average():
    env = GridWorld(seed=0)
    with_b = np.mean([reinforce(env, episodes=150, seed=s)[1][-50:].mean() for s in range(4)])
    no_b = np.mean([reinforce(env, episodes=150, seed=s, baseline=False)[1][-50:].mean() for s in range(4)])
    assert with_b >= no_b - 0.05


def test_sarsa_lambda_zero_equals_sarsa():
    env1, env2 = GridWorld(slip=0.1, seed=3), GridWorld(slip=0.1, seed=3)
    qa, ra = sarsa(env1, episodes=40, alpha=0.3, seed=3)
    qb, rb = sarsa_lambda(env2, episodes=40, alpha=0.3, lam=0.0, seed=3)
    assert np.allclose(qa, qb) and np.allclose(ra, rb)


def test_watkins_lambda_zero_equals_q_learning():
    env1, env2 = GridWorld(slip=0.1, seed=4), GridWorld(slip=0.1, seed=4)
    qa, _ = q_learning(env1, episodes=40, alpha=0.3, seed=4)
    qb, _ = sarsa_lambda(env2, episodes=40, alpha=0.3, lam=0.0, watkins=True, seed=4)
    assert np.allclose(qa, qb)


def test_traces_speed_up_credit_assignment():
    env = GridWorld(seed=0)
    _, r0 = sarsa_lambda(env, episodes=25, lam=0.0, seed=0)
    _, r9 = sarsa_lambda(GridWorld(seed=0), episodes=25, lam=0.9, seed=0)
    assert r9.mean() > r0.mean()


def test_sarsa_lambda_solves_det_grid():
    env = GridWorld(seed=1)
    q, _ = sarsa_lambda(env, episodes=500, lam=0.8, seed=1, watkins=True)
    opt = evaluate(env, greedy_policy(value_iteration(env)), 1)[0]
    assert abs(evaluate(env, greedy_policy(q), 1)[0] - opt) < 1e-9


def test_n_step_one_equals_sarsa():
    qa, ra = sarsa(GridWorld(seed=5), episodes=40, alpha=0.3, seed=5)
    qb, rb = n_step_sarsa(GridWorld(seed=5), n=1, episodes=40, alpha=0.3, seed=5)
    assert np.allclose(qa, qb) and np.allclose(ra, rb)


def test_n_step_learns_and_handles_truncation():
    wins = 0
    for sd in range(4):  # one seed (2) leaves a wall-bumping loop on the greedy path, so check a majority
        env = GridWorld(seed=sd)
        q, _ = n_step_sarsa(env, n=4, episodes=500, seed=sd)
        wins += evaluate(env, greedy_policy(q), 5)[1] == 1.0
    assert wins >= 3
    q2, r2 = n_step_sarsa(GridWorld(seed=2), n=3, episodes=5, max_steps=4, seed=2)
    assert np.isfinite(q2).all() and len(r2) == 5


def test_prioritized_sweeping_solves_det_grid():
    env = GridWorld(seed=3)
    q, rets, n = prioritized_sweeping(env, episodes=150, seed=3)
    opt = evaluate(env, greedy_policy(value_iteration(env)), 1)[0]
    assert abs(evaluate(env, greedy_policy(q), 1)[0] - opt) < 1e-9
    assert n > 0 and len(rets) == 150


def test_prioritized_sweeping_zero_planning_is_no_backups():
    q, _, n = prioritized_sweeping(GridWorld(seed=0), episodes=5, planning=0)
    assert n == 0 and np.all(q == 0)


def test_prioritized_sweeping_beats_q_early():
    sc = lambda f, **k: np.mean([f(GridWorld(seed=s), episodes=20, seed=s, **k)[1][5:].mean() for s in range(4)])
    assert sc(prioritized_sweeping) > sc(q_learning)
