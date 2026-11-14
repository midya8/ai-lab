"""DQN in plain numpy: one-hidden-layer MLP on coordinate features, uniform replay, target network."""
import numpy as np


def coord_features(env):
    """Row/col scaled to [0,1] plus one-hot-free bias; shape (n_states, 3)."""
    phi = np.zeros((env.n_states, 3))
    for r in range(env.h):
        for c in range(env.w):
            phi[env.sid((r, c))] = (r / max(env.h - 1, 1), c / max(env.w - 1, 1), 1.0)
    return phi


def onehot_features(env):
    return np.eye(env.n_states)


class QNet:
    def __init__(self, n_in, n_out, hidden=32, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(1 / hidden), (hidden, n_out)); self.b2 = np.zeros(n_out)

    def forward(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h, h @ self.W2 + self.b2

    def copy_from(self, o):
        self.W1, self.b1, self.W2, self.b2 = (o.W1.copy(), o.b1.copy(), o.W2.copy(), o.b2.copy())

    def step(self, x, a, target, lr, weights=None):
        """One SGD step on 0.5*(Q(x,a)-target)^2 (Huber-clipped gradient), batch mean."""
        h, q = self.forward(x)
        n = len(x)
        raw = q[np.arange(n), a] - target
        err = np.clip(raw, -1.0, 1.0)
        if weights is not None:
            err = err * weights
        dq = np.zeros_like(q); dq[np.arange(n), a] = err / n
        dh = (dq @ self.W2.T) * (h > 0)
        self.W2 -= lr * h.T @ dq; self.b2 -= lr * dq.sum(0)
        self.W1 -= lr * x.T @ dh; self.b1 -= lr * dh.sum(0)
        return raw


class DuelQNet(QNet):
    """Dueling architecture: Q = V(s) + A(s,a) - mean_a A(s,a) on a shared ReLU trunk."""
    def __init__(self, n_in, n_out, hidden=32, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, hidden)); self.b1 = np.zeros(hidden)
        self.Wv = rng.normal(0, np.sqrt(1 / hidden), (hidden, 1)); self.bv = np.zeros(1)
        self.Wa = rng.normal(0, np.sqrt(1 / hidden), (hidden, n_out)); self.ba = np.zeros(n_out)

    def forward(self, x):
        h = np.maximum(x @ self.W1 + self.b1, 0)
        a = h @ self.Wa + self.ba
        return h, (h @ self.Wv + self.bv) + a - a.mean(1, keepdims=True)

    def copy_from(self, o):
        for k in ("W1", "b1", "Wv", "bv", "Wa", "ba"):
            setattr(self, k, getattr(o, k).copy())

    def step(self, x, a, target, lr, weights=None):
        h, q = self.forward(x)
        n = len(x)
        raw = q[np.arange(n), a] - target
        err = np.clip(raw, -1.0, 1.0)
        if weights is not None:
            err = err * weights
        dq = np.zeros_like(q); dq[np.arange(n), a] = err / n
        dv = dq.sum(1, keepdims=True)
        da = dq - dq.mean(1, keepdims=True)
        dh = (dv @ self.Wv.T + da @ self.Wa.T) * (h > 0)
        self.Wv -= lr * h.T @ dv; self.bv -= lr * dv.sum(0)
        self.Wa -= lr * h.T @ da; self.ba -= lr * da.sum(0)
        self.W1 -= lr * x.T @ dh; self.b1 -= lr * dh.sum(0)
        return raw


def double_q_target(online, target, x):
    """Double DQN: the online net picks the action, the target net scores it."""
    a = online.forward(x)[1].argmax(1)
    return target.forward(x)[1][np.arange(len(x)), a]


def dqn(env, episodes=300, gamma=0.97, lr=0.2, eps_start=1.0, eps_end=0.05, eps_decay=0.99,
        batch=32, buffer=5000, target_every=50, hidden=64, seed=0, max_steps=200,
        use_target=True, use_replay=True, features="onehot", double=False,
        prioritized=False, per_alpha=0.6, per_beta=0.4, per_eps=1e-3, dueling=False):
    """Returns (net, phi, returns). Greedy policy: argmax over net.forward(phi)[1]."""
    rng = np.random.default_rng(seed)
    phi = coord_features(env) if features == "coords" else onehot_features(env)
    Net = DuelQNet if dueling else QNet
    net, tgt = Net(phi.shape[1], env.n_actions, hidden, rng), Net(phi.shape[1], env.n_actions, hidden, rng)
    tgt.copy_from(net)
    buf, prio, steps, rets = [], [], 0, []
    eps = eps_start
    for _ in range(episodes):
        s, tot = env.reset(), 0.0
        for _ in range(max_steps):
            if rng.random() < eps:
                a = int(rng.integers(env.n_actions))
            else:
                a = int(net.forward(phi[s:s + 1])[1].argmax())
            s2, r, done = env.step(a)
            tot += r
            tr = (s, a, r, s2, done)
            buf.append(tr)
            prio.append(max(prio, default=1.0))  # new transitions get max priority
            if len(buf) > buffer:
                buf.pop(0); prio.pop(0)
            idx, w = None, None
            if use_replay and len(buf) >= batch:
                if prioritized:
                    p = np.asarray(prio) ** per_alpha; p /= p.sum()
                    idx = rng.choice(len(buf), size=batch, p=p)
                    w = (len(buf) * p[idx]) ** (-per_beta); w /= w.max()
                else:
                    idx = rng.integers(len(buf), size=batch)
                batch_tr = [buf[i] for i in idx]
            else:
                batch_tr = [tr]
            S, A, R, S2, D = (np.array(z) for z in zip(*batch_tr))
            ref = tgt if use_target else net
            nxt = double_q_target(net, ref, phi[S2]) if double else ref.forward(phi[S2])[1].max(1)
            td = net.step(phi[S], A, R + gamma * nxt * (~D.astype(bool)), lr, w)
            if prioritized and idx is not None:
                for i, e in zip(idx, td):
                    prio[i] = abs(float(e)) + per_eps
            steps += 1
            if use_target and steps % target_every == 0:
                tgt.copy_from(net)
            s = s2
            if done:
                break
        eps = max(eps_end, eps * eps_decay)
        rets.append(tot)
    return net, phi, np.array(rets)


def dqn_policy(net, phi):
    return net.forward(phi)[1].argmax(1)
