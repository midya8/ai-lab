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
