"""Play players against each other and score results."""
import random
from .minimax import best_moves


def play_game(game, p1, p2):
    """p1 moves as +1, p2 as -1. Returns winner (+1/-1/0)."""
    s = game.initial()
    while not game.is_terminal(s):
        s = game.play(s, (p1 if s[1] == 1 else p2)(s))
    return game.winner(s)


def random_player(game, seed=0):
    rng = random.Random(seed)
    return lambda s: rng.choice(game.legal_moves(s))


def perfect_player(game, seed=0):
    rng = random.Random(seed)
    return lambda s: rng.choice(best_moves(game, s))


def match(game, a, b, games=20):
    """a vs b, alternating colours. Returns dict of a's wins/draws/losses."""
    res = {"wins": 0, "draws": 0, "losses": 0}
    for i in range(games):
        if i % 2 == 0:
            w = play_game(game, a, b); a_side = 1
        else:
            w = play_game(game, b, a); a_side = -1
        if w == 0: res["draws"] += 1
        elif w == a_side: res["wins"] += 1
        else: res["losses"] += 1
    return res
