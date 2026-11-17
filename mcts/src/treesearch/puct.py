"""PUCT search (AlphaZero-style selection) with a pluggable policy prior and value function.

No network is learned here: `prior_fn(game, state) -> {move: p}` and
`value_fn(game, state) -> float in [-1, 1]` (absolute: +1 means player +1 wins) are
plain callables, so a learned model can be dropped in later. Built-in helpers:
`uniform_prior`, `rollout_value` (random playout), `line_heuristic_value` (open-line count).
"""
from __future__ import annotations
import math
import random


def uniform_prior(game, state):
    moves = game.legal_moves(state)
    return {m: 1.0 / len(moves) for m in moves}


def rollout_value(game, state, rng=None, n=1):
    rng = rng or random.Random(0)
    tot = 0
    for _ in range(n):
        s = state
        while not game.is_terminal(s):
            s = game.play(s, rng.choice(game.legal_moves(s)))
        tot += game.winner(s)
    return tot / n


def _windows(game):
    R, C, k = game.rows, game.cols, game.k
    out = []
    for r in range(R):
        for c in range(C):
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                er, ec = r + dr * (k - 1), c + dc * (k - 1)
                if 0 <= er < R and 0 <= ec < C:
                    out.append([(r + dr * i) * C + c + dc * i for i in range(k)])
    return out


def line_heuristic_value(game, state):
    """Hand-written evaluation: sum over k-windows containing only one colour of (count/k)^2,
    squashed with tanh. Not learned and not tuned; a weak, fast static evaluator."""
    board, _ = state
    w = game.winner(state)
    if w:
        return float(w)
    score = 0.0
    for win in _windows(game):
        vals = [board[i] for i in win]
        if 1 in vals and -1 in vals:
            continue
        cnt = sum(vals)
        score += (abs(cnt) / game.k) ** 2 * (1 if cnt > 0 else -1 if cnt < 0 else 0)
    return math.tanh(score / 4.0)


class _N:
    __slots__ = ("state", "children", "p", "n", "w", "expanded")

    def __init__(self, state, p):
        self.state, self.p, self.n, self.w = state, p, 0, 0.0
        self.children = {}
        self.expanded = False


def puct_search(game, state, iterations=200, c_puct=1.5, prior_fn=uniform_prior,
                value_fn=None, rng=None):
    """Return (best_move, visit_counts). Value of a node is from the view of the player
    who moved into it (w/n in [-1,1]); selection maximises Q + c*P*sqrt(N)/(1+n)."""
    rng = rng or random.Random(0)
    value_fn = value_fn or (lambda g, s: rollout_value(g, s, rng))
    moves = game.legal_moves(state)
    if not moves:
        raise ValueError("no legal moves")
    root = _N(state, 1.0)

    def expand(node):
        node.expanded = True
        if game.is_terminal(node.state):
            return
        pri = prior_fn(game, node.state)
        z = sum(pri.get(m, 0.0) for m in game.legal_moves(node.state)) or 1.0
        for m in game.legal_moves(node.state):
            node.children[m] = _N(game.play(node.state, m), pri.get(m, 0.0) / z)

    for _ in range(iterations):
        node, path = root, [root]
        while node.expanded and node.children:
            sq = math.sqrt(max(node.n, 1))
            best, bs = None, -1e18
            for m, ch in node.children.items():
                q = ch.w / ch.n if ch.n else 0.0
                s = q + c_puct * ch.p * sq / (1 + ch.n)
                if s > bs:
                    best, bs = ch, s
            node = best
            path.append(node)
        if not node.expanded:
            expand(node)
        v = float(game.winner(node.state)) if game.is_terminal(node.state) else value_fn(game, node.state)
        for nd in path:
            mover = -nd.state[1]  # who moved into nd
            nd.n += 1
            if nd is not root:
                nd.w += v * mover
    counts = {m: ch.n for m, ch in root.children.items()}
    best = max(counts, key=lambda m: (counts[m], -m))
    return best, counts


class PUCT:
    def __init__(self, game, iterations=200, c_puct=1.5, prior_fn=uniform_prior, value_fn=None, seed=0):
        self.game, self.iterations, self.c = game, iterations, c_puct
        self.prior_fn, self.value_fn = prior_fn, value_fn
        self.rng = random.Random(seed)

    def __call__(self, state):
        return puct_search(self.game, state, self.iterations, self.c, self.prior_fn,
                           self.value_fn, self.rng)[0]
