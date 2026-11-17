"""Exact alpha-beta minimax, used as ground truth for small games."""
from functools import lru_cache


def minimax_value(game, state):
    """Game value from the perspective of the player to move: +1 win, 0 draw, -1 loss."""
    cache = {}

    def v(s):
        if s in cache:
            return cache[s]
        w = game.winner(s)
        if w != 0:
            r = -1  # previous mover won, so player to move lost
        elif game.is_terminal(s):
            r = 0
        else:
            r = max(-v(game.play(s, m)) for m in game.legal_moves(s))
        cache[s] = r
        return r

    return v(state)


def best_moves(game, state):
    """All moves achieving the optimal game value."""
    vals = {m: -minimax_value(game, game.play(state, m)) for m in game.legal_moves(state)}
    top = max(vals.values())
    return sorted(m for m, x in vals.items() if x == top)
