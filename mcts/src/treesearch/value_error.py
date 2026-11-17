"""How well does UCT estimate the *value* of a position (not just the move)?

Move accuracy can hide a badly calibrated value. We read the value off the most-visited
child (its mean reward for the mover, mapped to [-1, 1]) and compare it with the exact
minimax value, as a function of the iteration budget.
"""
from __future__ import annotations
import random
from .mcts import uct_search
from .minimax import minimax_value


def estimate_value(game, state, iterations=500, c=1.4, seed=0):
    """UCT estimate of the game value for the player to move, in [-1, 1].

    Rewards are win=1, draw=0.5, loss=0 for the mover into each node, so the most-visited
    child's w/n is the mover's expected score; map it with 2*x-1. Returns None for a
    single-move or terminal state where no tree is built (use the exact value there).
    """
    moves = game.legal_moves(state)
    if len(moves) < 2:
        return None
    _, counts, root = uct_search(game, state, iterations, c, random.Random(seed), return_root=True)
    ch = max(root.children, key=lambda x: x.n)
    return 2.0 * ch.w / ch.n - 1.0


def value_error_curve(game, states, budgets=(25, 100, 400), c=1.4, seed=0):
    """For each budget: mean absolute error, and sign accuracy (estimate rounds to exact value
    within 1/3 of a unit) vs exact minimax over `states`. Returns list of dicts."""
    exact = {s: minimax_value(game, s) for s in states}
    out = []
    for b in budgets:
        errs, ok, n = [], 0, 0
        for i, s in enumerate(states):
            v = estimate_value(game, s, b, c, seed + i)
            if v is None:
                continue
            e = abs(v - exact[s])
            errs.append(e)
            ok += e < 1 / 3
            n += 1
        out.append({"iterations": b, "n": n, "mae": sum(errs) / max(n, 1), "within_third": ok / max(n, 1)})
    return out
