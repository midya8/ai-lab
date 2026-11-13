"""AlphaZero-lite: a small numpy policy/value MLP trained purely from PUCT self-play.

No human data and no minimax labels are used for training. The net maps a board (from the view
of the player to move) to a masked-softmax move prior and a value in [-1, 1]; `net.prior_fn` and
`net.value_fn` plug straight into `puct_search`. Intended for small boards (tic-tac-toe and
friends); it is a teaching-scale implementation, not a general AlphaZero.
"""
from __future__ import annotations
import math
import random
import numpy as np
from .puct import puct_search


def encode(game, state):
    board, player = state
    b = np.asarray(board, dtype=np.float64)
    return np.concatenate([(b == player), (b == -player)]).astype(np.float64)


class PVNet:
    def __init__(self, game, hidden=64, seed=0):
        self.game = game
        self.n = game.rows * game.cols
        rng = np.random.default_rng(seed)
        d = 2 * self.n
        self.p = {
            "W1": rng.normal(0, 1 / math.sqrt(d), (d, hidden)), "b1": np.zeros(hidden),
            "Wp": rng.normal(0, 0.1, (hidden, self.n)), "bp": np.zeros(self.n),
            "Wv": rng.normal(0, 0.1, (hidden, 1)), "bv": np.zeros(1),
        }
        self._m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._s = {k: np.zeros_like(v) for k, v in self.p.items()}
        self._t = 0

    def forward(self, X, mask):
        p = self.p
        h = np.tanh(X @ p["W1"] + p["b1"])
        logits = h @ p["Wp"] + p["bp"]
        logits = np.where(mask, logits, -1e9)
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits) * mask
        pi = e / e.sum(axis=1, keepdims=True)
        v = np.tanh(h @ p["Wv"] + p["bv"])[:, 0]
        return h, pi, v

    def _mask(self, states):
        m = np.zeros((len(states), self.n), dtype=bool)
        for i, s in enumerate(states):
            for mv in self.game.legal_moves(s):
                m[i, mv] = True
        return m

    def predict(self, state):
        mask = self._mask([state])
        if not mask.any():
            return None, 0.0
        _, pi, v = self.forward(encode(self.game, state)[None], mask)
        return pi[0], float(v[0])

    # PUCT plug-ins (move indices equal cell indices on non-gravity boards)
    def prior_fn(self, game, state):
        pi, _ = self.predict(state)
        return {m: float(pi[m]) for m in game.legal_moves(state)}

    def value_fn(self, game, state):
        _, v = self.predict(state)
        return v * state[1]  # absolute: +1 means player +1 wins

    def train_batch(self, X, masks, target_pi, target_z, lr=0.01, l2=1e-4):
        """One Adam step on mean cross-entropy(pi) + mean squared error(z). Returns (policy_loss, value_loss)."""
        p = self.p
        B = len(X)
        h, pi, v = self.forward(X, masks)
        lp = -np.sum(target_pi * np.log(pi + 1e-12)) / B
        lv = np.mean((v - target_z) ** 2)
        dlog = (pi - target_pi) / B
        dv_pre = (2 * (v - target_z) / B * (1 - v ** 2))[:, None]
        g = {
            "Wp": h.T @ dlog, "bp": dlog.sum(0),
            "Wv": h.T @ dv_pre, "bv": dv_pre.sum(0),
        }
        dh = dlog @ p["Wp"].T + dv_pre @ p["Wv"].T
        dpre = dh * (1 - h ** 2)
        g["W1"], g["b1"] = X.T @ dpre, dpre.sum(0)
        self._t += 1
        for k in p:
            grad = g[k] + l2 * p[k]
            self._m[k] = 0.9 * self._m[k] + 0.1 * grad
            self._s[k] = 0.999 * self._s[k] + 0.001 * grad ** 2
            mh = self._m[k] / (1 - 0.9 ** self._t)
            sh = self._s[k] / (1 - 0.999 ** self._t)
            p[k] -= lr * mh / (np.sqrt(sh) + 1e-8)
        return float(lp), float(lv)

    def save(self, path):
        np.savez(path, **self.p)

    def load(self, path):
        d = np.load(path)
        for k in self.p:
            self.p[k] = d[k]


def self_play_game(game, net, iterations, rng, temp_moves=4, noise=0.25):
    """Play one game with net-guided PUCT; return [(state, visit_policy)], winner.

    Exploration: root move sampled from visit counts for the first `temp_moves` plies, and with
    probability `noise` a uniformly random legal move is played instead (a simple stand-in for
    Dirichlet noise, which the search does not support)."""
    s = game.initial()
    rec = []
    ply = 0
    while not game.is_terminal(s):
        _, counts = puct_search(game, s, iterations, prior_fn=net.prior_fn, value_fn=net.value_fn, rng=rng)
        tot = sum(counts.values())
        pol = np.zeros(net.n)
        for m, c in counts.items():
            pol[m] = c / tot
        rec.append((s, pol))
        moves = list(counts)
        if rng.random() < noise:
            mv = rng.choice(moves)
        elif ply < temp_moves:
            mv = rng.choices(moves, weights=[counts[m] for m in moves])[0]
        else:
            mv = max(moves, key=lambda m: (counts[m], -m))
        s = game.play(s, mv)
        ply += 1
    return rec, game.winner(s)


def train(game, net, rounds=30, games_per_round=8, iterations=25, epochs=20, lr=0.01, seed=0,
          buffer_size=4000, log=None):
    rng = random.Random(seed)
    buf = []
    nprng = np.random.default_rng(seed)
    for r in range(rounds):
        for _ in range(games_per_round):
            rec, w = self_play_game(game, net, iterations, rng)
            for s, pol in rec:
                buf.append((s, pol, float(w * s[1])))  # outcome for the player to move at s
        buf = buf[-buffer_size:]
        X = np.stack([encode(game, s) for s, _, _ in buf])
        M = net._mask([s for s, _, _ in buf])
        P = np.stack([p for _, p, _ in buf])
        Z = np.array([z for _, _, z in buf])
        for _ in range(epochs):
            idx = nprng.permutation(len(buf))[:256]
            lp, lv = net.train_batch(X[idx], M[idx], P[idx], Z[idx], lr=lr)
        if log:
            log(r, lp, lv, len(buf))
    return net
