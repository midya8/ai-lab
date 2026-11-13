import numpy as np
from rlgrid import GridWorld, evaluate
from rlgrid.dqn import dqn, dqn_policy, QNet


def test_qnet_regresses_single_target():
    net = QNet(3, 4, 16, np.random.default_rng(0))
    x = np.array([[0.2, 0.5, 1.0]])
    for _ in range(300):
        net.step(x, np.array([2]), np.array([0.7]), 0.05)
    assert abs(net.forward(x)[1][0, 2] - 0.7) < 0.05


def test_gradient_matches_finite_difference():
    net = QNet(3, 4, 8, np.random.default_rng(1))
    x = np.random.default_rng(2).normal(size=(5, 3)); a = np.array([0, 1, 2, 3, 0]); t = np.zeros(5) + 0.1
    def loss():
        q = net.forward(x)[1][np.arange(5), a]
        return 0.5 * np.mean((q - t) ** 2)
    w0 = net.W1[0, 0]; eps = 1e-6
    net.W1[0, 0] = w0 + eps; lp = loss(); net.W1[0, 0] = w0 - eps; lm = loss(); net.W1[0, 0] = w0
    num = (lp - lm) / (2 * eps)
    snap = net.W1.copy(); net.step(x, a, t, 1.0)
    analytic = snap[0, 0] - net.W1[0, 0]
    assert abs(num - analytic) < 1e-5


def test_dqn_reaches_goal_on_small_grid():
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    net, phi, rets = dqn(env, episodes=250, seed=0)
    ret, win = evaluate(env, dqn_policy(net, phi), episodes=5)
    assert win == 1.0 and rets[-20:].mean() > rets[:20].mean()


def test_deterministic_given_seed():
    env1, env2 = GridWorld(layout=["S.G"], seed=0), GridWorld(layout=["S.G"], seed=0)
    r1 = dqn(env1, episodes=20, seed=3)[2]; r2 = dqn(env2, episodes=20, seed=3)[2]
    assert np.array_equal(r1, r2)
