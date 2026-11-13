"""UCT with a transposition table: statistics are shared by identical states.

Plain UCT builds a tree, so the same position reached by different move orders
is learned separately. Here each state owns one (visits, reward, untried) record
and a simulation walks a path through the implied DAG, updating every record on
the path (the "UCT with transpositions" scheme of Childs et al., 2008).
"""
from __future__ import annotations
import math
import random


class _Rec:
    __slots__ = ("n", "w", "untried")

    def __init__(self, moves):
        self.n = 0
        self.w = 0.0  # reward for the player who moved INTO this state
        self.untried = list(moves)


def tt_uct_search(game, state, iterations=500, c=1.4, rng=None, return_table=False):
    """Return (best_move, root visit counts per move) [, table]."""
    rng = rng or random.Random(0)
    moves = game.legal_moves(state)
    if not moves:
        raise ValueError("no legal moves")
    if len(moves) == 1 and not return_table:
        return moves[0], {moves[0]: iterations}
    table = {}

    def rec(s):
        r = table.get(s)
        if r is None:
            r = table[s] = _Rec(game.legal_moves(s))
        return r

    root_visits = {m: 0 for m in moves}
    for _ in range(iterations):
        s = state
        path = [s]
        first = None
        while True:
            r = rec(s)
            if game.is_terminal(s):
                break
            if r.untried:
                m = r.untried.pop(rng.randrange(len(r.untried)))
                s = game.play(s, m)
                path.append(s)
                if first is None:
                    first = m
                break
            log_n = math.log(max(r.n, 1))
            best, best_u = None, -1.0
            for m in game.legal_moves(s):
                ch = table[game.play(s, m)] if game.play(s, m) in table else None
                if ch is None or ch.n == 0:
                    u = float("inf")
                else:
                    u = ch.w / ch.n + c * math.sqrt(log_n / ch.n)
                if u > best_u:
                    best, best_u = m, u
            if first is None:
                first = best
            s = game.play(s, best)
            path.append(s)
        rec(s)
        winner = game.winner(s)
        if first is not None:
            root_visits[first] += 1
        for i, ps in enumerate(path):
            pr = rec(ps)
            if i > 0:
                mover = -ps[1]
                pr.w += 1.0 if winner == mover else (0.5 if winner == 0 else 0.0)
            pr.n += 1
    best = max(root_visits, key=lambda m: (root_visits[m], -m))
    return (best, root_visits, table) if return_table else (best, root_visits)


class TTMCTS:
    """Callable player: TTMCTS(game, iterations)(state) -> move."""

    def __init__(self, game, iterations=500, c=1.4, seed=0):
        self.game, self.iterations, self.c = game, iterations, c
        self.rng = random.Random(seed)

    def __call__(self, state):
        return tt_uct_search(self.game, state, self.iterations, self.c, self.rng)[0]
