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
            elif mode == "esarsa":
                greedy = (q[s2] == q[s2].max())
                pi = np.full(q.shape[1], eps / q.shape[1]) + (1 - eps) * greedy / greedy.sum()
                target = r + gamma * float(pi @ q[s2])
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


def expected_sarsa(env, episodes=500, alpha=0.5, gamma=0.97, eps=0.1, seed=0, max_steps=200):
    """On-policy target = expectation over the eps-greedy policy (lower variance than SARSA)."""
    return _run(env, episodes, alpha, gamma, eps, seed, max_steps, "esarsa")


def double_q(env, episodes=500, alpha=0.5, gamma=0.97, eps=0.1, seed=0, max_steps=200):
    """Double Q-learning (van Hasselt 2010). Returns (qA+qB)/2 and episode returns."""
    rng = np.random.default_rng(seed)
    qa = np.zeros((env.n_states, env.n_actions)); qb = qa.copy()
    returns = []
    for _ in range(episodes):
        s = env.reset(); total = 0.0
        for _ in range(max_steps):
            a = _eps_greedy(qa + qb, s, eps, rng)
            s2, r, done = env.step(a)
            total += r
            x, y = (qa, qb) if rng.random() < 0.5 else (qb, qa)
            if done:
                t = r
            else:
                best = int(rng.choice(np.flatnonzero(x[s2] == x[s2].max())))
                t = r + gamma * y[s2, best]
            x[s, a] += alpha * (t - x[s, a])
            s = s2
            if done:
                break
        returns.append(total)
    return (qa + qb) / 2, np.array(returns)


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


def reinforce(env, episodes=500, lr=0.2, gamma=0.97, seed=0, max_steps=200, baseline=True, lr_v=0.2):
    """REINFORCE (Williams 1992) with a tabular softmax policy.

    With baseline=True a per-state value table V(s) is learned by Monte-Carlo
    regression and subtracted from the return (variance reduction, same
    expected gradient). Returns (logits, episode_returns); greedy_policy(logits)
    gives the argmax policy.
    """
    rng = np.random.default_rng(seed)
    theta = np.zeros((env.n_states, env.n_actions))
    v = np.zeros(env.n_states)
    returns = []
    for _ in range(episodes):
        s = env.reset()
        traj, total = [], 0.0
        for _ in range(max_steps):
            z = theta[s] - theta[s].max()
            p = np.exp(z) / np.exp(z).sum()
            a = int(rng.choice(len(p), p=p))
            s2, r, done = env.step(a)
            traj.append((s, a, r))
            total += r
            s = s2
            if done:
                break
        g = 0.0
        for s_t, a_t, r_t in reversed(traj):
            g = r_t + gamma * g
            adv = g - (v[s_t] if baseline else 0.0)
            z = theta[s_t] - theta[s_t].max()
            p = np.exp(z) / np.exp(z).sum()
            grad = -p
            grad[a_t] += 1.0
            theta[s_t] += lr * adv * grad
            if baseline:
                v[s_t] += lr_v * (g - v[s_t])
        returns.append(total)
    return theta, np.array(returns)


def sarsa_lambda(env, episodes=500, alpha=0.3, gamma=0.97, eps=0.1, lam=0.8, seed=0,
                 max_steps=200, watkins=False, trace="replacing"):
    """SARSA(lambda) with eligibility traces; watkins=True gives Watkins's Q(lambda)
    (off-policy: traces are cut after an exploratory action). lam=0 equals one-step SARSA
    (or Q-learning when watkins=True)."""
    assert trace in ("replacing", "accumulating") and 0.0 <= lam <= 1.0
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        e = np.zeros_like(q)
        s = env.reset()
        a = _eps_greedy(q, s, eps, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            total += r
            a2 = _eps_greedy(q, s2, eps, rng)
            if done:
                target = r
            elif watkins:
                target = r + gamma * q[s2].max()
            else:
                target = r + gamma * q[s2, a2]
            delta = target - q[s, a]
            if trace == "replacing":
                e[s] = 0.0
                e[s, a] = 1.0
            else:
                e[s, a] += 1.0
            q += alpha * delta * e
            e *= gamma * lam
            if watkins and not done and q[s2, a2] < q[s2].max():
                e[:] = 0.0  # exploratory action: cut the trace
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    return q, np.array(returns)


def n_step_sarsa(env, n=4, episodes=500, alpha=0.3, gamma=0.97, eps=0.1, seed=0, max_steps=200):
    """On-policy n-step SARSA (Sutton & Barto 7.2)."""
    assert n >= 1
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for _ in range(episodes):
        s = env.reset()
        a = _eps_greedy(q, s, eps, rng)
        S, A, R = [s], [a], [0.0]
        T, t, total, term = max_steps, 0, 0.0, False
        while True:
            if t < T:
                s2, r, done = env.step(A[t])
                total += r
                S.append(s2); R.append(r)
                a2 = _eps_greedy(q, s2, eps, rng)  # always drawn so n=1 matches sarsa() RNG-for-RNG
                if done or t + 1 >= max_steps:
                    T = t + 1
                    term = done
                else:
                    A.append(a2)
            tau = t - n + 1
            if tau >= 0:
                end = min(tau + n, T)
                G = sum(gamma ** (i - tau - 1) * R[i] for i in range(tau + 1, end + 1))
                if tau + n < T:
                    G += gamma ** n * q[S[tau + n], A[tau + n]]
                elif not term:  # time-limit truncation: bootstrap
                    G += gamma ** (end - tau) * q[S[end]].max()
                q[S[tau], A[tau]] += alpha * (G - q[S[tau], A[tau]])
            if tau == T - 1:
                break
            t += 1
        returns.append(total)
    return q, np.array(returns)


def prioritized_sweeping(env, episodes=500, alpha=0.5, gamma=0.97, eps=0.1, seed=0, max_steps=200,
                         planning=10, theta=1e-4):
    """Prioritized sweeping (Moore & Atkeson 1993; Sutton & Barto 8.4) with a deterministic last-seen model.
    Planning backups are taken from a max-priority queue of |TD error| > theta and propagate to the
    predecessors of each updated state. Returns (q, episode_returns, n_backups) where n_backups counts every
    Q update (real + planning), so sample/compute efficiency can be compared with dyna_q."""
    import heapq
    rng = np.random.default_rng(seed)
    q = np.zeros((env.n_states, env.n_actions))
    model, preds, returns, backups = {}, {}, [], 0
    pq, best = [], {}  # heap of (-priority, counter, s, a); best[(s,a)] = latest priority
    counter = 0

    def push(s, a, p):
        nonlocal counter
        if p > theta and p > best.get((s, a), 0.0):
            best[(s, a)] = p
            counter += 1
            heapq.heappush(pq, (-p, counter, s, a))

    def td(s, a):
        r, s2, d = model[(s, a)]
        return (r if d else r + gamma * q[s2].max()) - q[s, a]

    for _ in range(episodes):
        s = env.reset(); total = 0.0
        for _ in range(max_steps):
            a = _eps_greedy(q, s, eps, rng)
            s2, r, done = env.step(a)
            total += r
            model[(s, a)] = (r, s2, done)
            preds.setdefault(s2, set()).add((s, a))
            push(s, a, abs(td(s, a)))
            for _ in range(planning):
                if not pq:
                    break
                negp, _, ps, pa = heapq.heappop(pq)
                if best.get((ps, pa), 0.0) != -negp:
                    continue  # stale entry
                best.pop((ps, pa))
                q[ps, pa] += alpha * td(ps, pa)
                backups += 1
                for qs, qa in preds.get(ps, ()):
                    push(qs, qa, abs(td(qs, qa)))
            s = s2
            if done:
                break
        returns.append(total)
    return q, np.array(returns), backups


def make_features(env, kind="onehot", rbf_sigma=1.0):
    """State -> feature matrix (n_states, d). kinds: onehot (tabular), coords
    (bias, row, col, normalised: cannot represent walls), rbf (Gaussian bumps on
    every cell, width rbf_sigma in cells: generalises across neighbours)."""
    n = env.n_states
    rc = np.array([(s // env.w, s % env.w) for s in range(n)], dtype=float)
    if kind == "onehot":
        return np.eye(n)
    if kind == "coords":
        return np.column_stack([np.ones(n), rc[:, 0] / max(env.h - 1, 1), rc[:, 1] / max(env.w - 1, 1)])
    if kind == "rbf":
        d2 = ((rc[:, None, :] - rc[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2 * rbf_sigma ** 2))
    raise ValueError(f"unknown feature kind {kind!r}")


def actor_critic(env, episodes=500, alpha_pi=0.1, alpha_v=0.1, gamma=0.97, seed=0,
                 max_steps=200, features="onehot", rbf_sigma=1.0):
    """One-step actor-critic with LINEAR function approximation.

    Policy: softmax(phi(s) @ theta); critic: V(s) = phi(s) @ w, trained by TD(0).
    The TD error is the advantage estimate: theta += alpha_pi * delta * grad log pi.
    Updates are per step (online), unlike REINFORCE's per-episode Monte-Carlo.
    Returns (theta, w, episode_returns, phi); policy(s) = argmax(phi[s] @ theta).
    """
    rng = np.random.default_rng(seed)
    phi = make_features(env, features, rbf_sigma) if isinstance(features, str) else np.asarray(features, float)
    d = phi.shape[1]
    theta, w = np.zeros((d, env.n_actions)), np.zeros(d)
    returns = []
    for _ in range(episodes):
        s, total = env.reset(), 0.0
        for _ in range(max_steps):
            logits = phi[s] @ theta
            p = np.exp(logits - logits.max())
            p /= p.sum()
            a = int(rng.choice(len(p), p=p))
            s2, r, done = env.step(a)
            # bootstrap through time-limit truncation, not through true termination
            target = r + (0.0 if done else gamma * (phi[s2] @ w))
            delta = target - phi[s] @ w
            w += alpha_v * delta * phi[s]
            grad = -p
            grad[a] += 1.0
            theta += alpha_pi * delta * np.outer(phi[s], grad)
            total += r
            s = s2
            if done:
                break
        returns.append(total)
    return theta, w, np.array(returns), phi


def ac_policy(theta, phi):
    return (phi @ theta).argmax(axis=1)


def policy_evaluation(env, policy, gamma=0.97, tol=1e-10):
    """Exact V^pi of a fixed deterministic policy from the known model."""
    v = np.zeros(env.n_states)
    cells = [(r, c) for r in range(env.h) for c in range(env.w)
             if (r, c) not in env.walls and not env.terminal((r, c))]
    while True:
        delta = 0.0
        for pos in cells:
            s = env.sid(pos)
            new = 0.0
            for p, n in env.transitions(pos, int(policy[s])):
                fut = 0.0 if env.terminal(n) else v[env.sid(n)]
                new += p * (env.reward(n) + gamma * fut)
            delta = max(delta, abs(new - v[s]))
            v[s] = new
        if delta < tol:
            return v


def policy_regret(env, policy, gamma=0.97):
    """Exact per-state regret V* - V^pi (>= 0) over non-terminal free cells,
    plus its mean; no sampling noise, unlike evaluate()."""
    vstar = value_iteration(env, gamma).max(axis=1)
    vpi = policy_evaluation(env, policy, gamma)
    states = [env.sid(p) for p in env.free_cells() if not env.terminal(p)]
    reg = np.array([vstar[s] - vpi[s] for s in states])
    return reg, float(reg.mean())


def regret_curve(env, algo, checkpoints=(25, 50, 100, 200, 400), seeds=(0, 1, 2), gamma=0.97, **kw):
    """Sample-efficiency curve: exact mean policy regret of the greedy policy
    after training `algo` (a q_learning-style fn returning (q, returns)) for
    each episode budget. Returns {budget: (mean over seeds, std over seeds)}.
    Each (budget, seed) is an independent run, so budgets are not nested."""
    out = {}
    for n in checkpoints:
        vals = []
        for sd in seeds:
            q = algo(env, episodes=n, seed=sd, gamma=gamma, **kw)[0]
            vals.append(policy_regret(env, greedy_policy(q), gamma)[1])
        out[int(n)] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def policy_iteration(env, gamma=0.97, tol=1e-10, max_iters=100):
    """Howard's policy iteration from the known model.
    Evaluates exactly, then improves greedily (ties keep the current action so
    it terminates). Returns (policy, V, n_iterations). Converges in a handful
    of sweeps, far fewer than value iteration's Bellman backups."""
    policy = np.zeros(env.n_states, dtype=int)
    free = [p for p in env.free_cells() if not env.terminal(p)]
    for it in range(1, max_iters + 1):
        v = policy_evaluation(env, policy, gamma, tol)
        stable = True
        for pos in free:
            s = env.sid(pos)
            qs = np.zeros(4)
            for a in range(4):
                for p, n in env.transitions(pos, a):
                    fut = 0.0 if env.terminal(n) else v[env.sid(n)]
                    qs[a] += p * (env.reward(n) + gamma * fut)
            best = int(policy[s])
            if qs.max() > qs[best] + 1e-9:
                policy[s] = int(qs.argmax())
                stable = False
        if stable:
            return policy, v, it
    return policy, v, max_iters


def expected_visits(env, policy):
    """Exact expected number of visits to every state per episode (start counts
    as one visit) under a fixed deterministic policy, from the known model.
    Solves (I - P^T) n = e_start over non-terminal states. Returns None when the
    policy never terminates from the start with probability 1 (it can loop)."""
    seen, stack = {env.start}, [env.start]  # only states reachable under the policy
    while stack:
        pos = stack.pop()
        for _, nxt in env.transitions(pos, int(policy[env.sid(pos)])):
            if not env.terminal(nxt) and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    cells = sorted(seen)
    idx = {env.sid(p): i for i, p in enumerate(cells)}
    n = len(cells)
    P = np.zeros((n, n))
    for pos in cells:
        for pr, nxt in env.transitions(pos, int(policy[env.sid(pos)])):
            if not env.terminal(nxt):
                P[idx[env.sid(pos)], idx[env.sid(nxt)]] += pr
    e = np.zeros(n)
    e[idx[env.sid(env.start)]] = 1.0
    try:
        sol = np.linalg.solve(np.eye(n) - P.T, e)
    except np.linalg.LinAlgError:
        return None
    if (sol < -1e-9).any() or not np.isfinite(sol).all():
        return None
    out = np.zeros(env.n_states)
    for s, i in idx.items():
        out[s] = sol[i]
    return out


def expected_episode_length(env, policy):
    """Exact expected steps to termination (None if the policy can loop forever)."""
    v = expected_visits(env, policy)
    return None if v is None else float(v.sum())
