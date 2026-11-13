"""Tabular RL: Q-learning, SARSA, Dyna-Q, plus value iteration as ground truth."""
import numpy as np


def _eps_greedy(q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(q.shape[1]))
    best = np.flatnonzero(q[s] == q[s].max())
    return int(rng.choice(best))


def _run(env, episodes, alpha, gamma, eps, seed, max_steps, mode, planning=0):
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    model, returns = {}, []
    for _ in range(episodes):
        s = env.reset()
        a = _eps_greedy(q, s, eps, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            total += r
            a2 = _eps_greedy(q, s2, eps, rng)
            if done:
                target = r
            elif mode == "sarsa":
                target = r + gamma * q[s2, a2]
            else:
                target = r + gamma * q[s2].max()
            q[s, a] += alpha * (target - q[s, a])
            if planning:
                model[(s, a)] = (r, s2, done)
                keys = list(model)
                for i in rng.integers(len(keys), size=planning):
                    ps, pa = keys[i]
                    pr, pn, pd = model[(ps, pa)]
                    t = pr if pd else pr + gamma * q[pn].max()
                    q[ps, pa] += alpha * (t - q[ps, pa])
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    return q, np.array(returns)


def q_learning(env, episodes=500, alpha=0.5, gamma=0.97, eps=0.1, seed=0, max_steps=200):
    return _run(env, episodes, alpha, gamma, eps, seed, max_steps, "q")


def sarsa(env, episodes=500, alpha=0.5, gamma=0.97, eps=0.1, seed=0, max_steps=200):
    return _run(env, episodes, alpha, gamma, eps, seed, max_steps, "sarsa")


def dyna_q(env, episodes=500, alpha=0.5, gamma=0.97, eps=0.1, seed=0, max_steps=200, planning=10):
    return _run(env, episodes, alpha, gamma, eps, seed, max_steps, "q", planning)


def value_iteration(env, gamma=0.97, tol=1e-10):
    """Exact Q* from the known model (handles slip)."""
    q = np.zeros((env.n_states, env.n_actions))
    cells = [(r, c) for r in range(env.h) for c in range(env.w) if (r, c) not in env.walls]
    while True:
        delta = 0.0
        for pos in cells:
            if env.terminal(pos):
                continue
            s = env.sid(pos)
            for a in range(4):
                v = 0.0
                for p, n in env.transitions(pos, a):
                    fut = 0.0 if env.terminal(n) else q[env.sid(n)].max()
                    v += p * (env.reward(n) + gamma * fut)
                delta = max(delta, abs(v - q[s, a]))
                q[s, a] = v
        if delta < tol:
            return q


def greedy_policy(q):
    return q.argmax(axis=1)


def evaluate(env, policy, episodes=100, max_steps=200):
    """Mean return and success rate of a fixed policy."""
    rets, wins = [], 0
    for _ in range(episodes):
        s = env.reset()
        tot = 0.0
        for _ in range(max_steps):
            s, r, done = env.step(int(policy[s]))
            tot += r
            if done:
                wins += env.pos in env.goals
                break
        rets.append(tot)
    return float(np.mean(rets)), wins / episodes
