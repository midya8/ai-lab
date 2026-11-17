"""Exact move-quality evaluation of a search player against minimax ground truth."""
import random
from .minimax import best_moves


def reachable_states(game, limit=None, seed=0):
    """All non-terminal states reachable from the start (BFS), optionally a seeded sample of `limit`."""
    seen = {game.initial()}
    frontier = [game.initial()]
    while frontier:
        nxt = []
        for s in frontier:
            if game.is_terminal(s):
                continue
            for m in game.legal_moves(s):
                t = game.play(s, m)
                if t not in seen:
                    seen.add(t)
                    nxt.append(t)
        frontier = nxt
    states = sorted(s for s in seen if not game.is_terminal(s))
    if limit is not None and limit < len(states):
        states = random.Random(seed).sample(states, limit)
    return states


def optimal_move_rate(game, choose, states):
    """Fraction of `states` where choose(state) is a minimax-optimal move.

    Returns (rate, blunders) where blunders lists (state, move) pairs that were not optimal.
    """
    bad = []
    for s in states:
        m = choose(s)
        if m not in best_moves(game, s):
            bad.append((s, m))
    return 1 - len(bad) / len(states), bad
