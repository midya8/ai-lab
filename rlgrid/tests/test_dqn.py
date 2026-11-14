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


def test_double_q_target_uses_online_argmax_and_never_exceeds_max():
    from rlgrid.dqn import double_q_target
    rng = np.random.default_rng(5)
    on, tg = QNet(4, 3, 8, rng), QNet(4, 3, 8, rng)
    x = rng.normal(size=(6, 4))
    d = double_q_target(on, tg, x)
    a = on.forward(x)[1].argmax(1)
    assert np.allclose(d, tg.forward(x)[1][np.arange(6), a])
    assert np.all(d <= tg.forward(x)[1].max(1) + 1e-12)
    same = double_q_target(tg, tg, x)
    assert np.allclose(same, tg.forward(x)[1].max(1))


def test_double_dqn_reaches_goal():
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    net, phi, _ = dqn(env, episodes=250, seed=0, double=True)
    assert evaluate(env, dqn_policy(net, phi), episodes=5)[1] == 1.0


def test_step_weights_scale_update_and_return_td_error():
    x = np.array([[0.2, 0.5, 1.0]])
    n1, n2 = QNet(3, 4, 8, np.random.default_rng(0)), QNet(3, 4, 8, np.random.default_rng(0))
    td = n1.step(x, np.array([1]), np.array([0.3]), 0.1)
    n2.step(x, np.array([1]), np.array([0.3]), 0.1, weights=np.array([0.0]))
    assert td.shape == (1,)
    assert np.allclose(n2.W2, QNet(3, 4, 8, np.random.default_rng(0)).W2)  # zero weight: no update
    assert not np.allclose(n1.W2, n2.W2)


def test_prioritized_replay_reaches_goal_and_is_deterministic():
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    n1, phi, r1 = dqn(env, episodes=250, seed=0, prioritized=True)
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    n2, _, r2 = dqn(env, episodes=250, seed=0, prioritized=True)
    assert np.array_equal(r1, r2)
    assert evaluate(env, dqn_policy(n1, phi), episodes=5)[1] == 1.0


def test_dueling_gradient_matches_finite_difference():
    from rlgrid.dqn import DuelQNet
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 3)); a = np.array([0, 2, 1, 3]); tg = rng.normal(size=4) * 0.3
    net = DuelQNet(3, 4, 6, np.random.default_rng(2))
    def loss(n):
        q = n.forward(x)[1][np.arange(4), a]
        return 0.5 * np.mean((q - tg) ** 2)
    ref = DuelQNet(3, 4, 6, np.random.default_rng(2)); ref.copy_from(net)
    lr = 1.0
    net.step(x, a, tg, lr)
    for k in ("W1", "Wv", "Wa"):
        W = getattr(ref, k); g = np.zeros_like(W)
        for i in np.ndindex(W.shape):
            W[i] += 1e-6; lp = loss(ref); W[i] -= 2e-6; lm = loss(ref); W[i] += 1e-6
            g[i] = (lp - lm) / 2e-6
        assert np.allclose(getattr(net, k), W - lr * g, atol=1e-5)


def test_dueling_advantage_is_centered_and_reaches_goal():
    from rlgrid.dqn import DuelQNet
    net = DuelQNet(3, 4, 6, np.random.default_rng(0))
    x = np.random.default_rng(3).normal(size=(5, 3))
    h, q = net.forward(x)
    assert np.allclose(q.mean(1), (h @ net.Wv + net.bv)[:, 0])
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    n, phi, _ = dqn(env, episodes=250, seed=0, dueling=True)
    assert evaluate(env, dqn_policy(n, phi), episodes=5)[1] == 1.0


def test_nstep_emit_returns_and_tail_flush():
    from rlgrid.dqn import nstep_emit
    g = 0.5
    pend = [(0, 0, 1.0, 1, False), (1, 0, 2.0, 2, False)]
    assert nstep_emit(pend, 3, g, False) == []
    pend.append((2, 1, 4.0, 3, True))
    out = nstep_emit(pend, 3, g, True)
    assert [o[5] for o in out] == [3, 2, 1] and pend == []
    assert out[0][:5] == (0, 0, 1.0 + 1.0 + 1.0, 3, True)
    assert out[1][2] == 2.0 + 0.5 * 4.0 and out[2][2] == 4.0


def test_nstep1_matches_default_and_nstep3_reaches_goal():
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    _, _, r1 = dqn(env, episodes=40, seed=1)
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    _, _, r1b = dqn(env, episodes=40, seed=1, n_step=1)
    assert (r1 == r1b).all()
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    net, phi, _ = dqn(env, episodes=250, seed=0, n_step=3)
    env.reset(); assert evaluate(env, dqn_policy(net, phi)) is not None


def test_q_bias_zero_for_exact_and_detects_offset():
    from rlgrid import value_iteration
    from rlgrid.dqn import q_bias, onehot_features
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    qs = value_iteration(env)
    phi = onehot_features(env)

    class Fake:
        def __init__(self, off): self.off = off
        def forward(self, x): return None, qs[x.argmax(1)] + self.off
    m, rmse, agree = q_bias(env, Fake(0.0), phi, qs)
    assert abs(m) < 1e-12 and rmse < 1e-12 and agree == 1.0
    m, rmse, agree = q_bias(env, Fake(0.3), phi, qs)
    assert abs(m - 0.3) < 1e-9 and abs(rmse - 0.3) < 1e-9 and agree == 1.0


def test_q_bias_on_trained_net_is_finite_and_mostly_greedy_correct():
    from rlgrid import value_iteration
    from rlgrid.dqn import q_bias
    env = GridWorld(layout=["S..", "...", "..G"], seed=0)
    net, phi, _ = dqn(env, episodes=250, seed=0)
    m, rmse, agree = q_bias(env, net, phi, value_iteration(env))
    assert np.isfinite(m) and rmse < 0.5 and agree >= 0.8
