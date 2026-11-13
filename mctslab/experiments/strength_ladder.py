"""Does more search help? UCT strength ladder on Connect-Four-style boards.

Usage: PYTHONPATH=src python experiments/strength_ladder.py [games] [rows cols k]
Prints a win-rate table: each budget vs random and vs the previous budget.
Draws count as half a win. Small samples: read as trends, not precise Elo.
"""
import sys, random
from mctslab import ConnectN, uct_search
from mctslab.arena import match, random_player


def uct_player(game, iters, seed):
    rng = random.Random(seed)
    return lambda s: uct_search(game, s, iterations=iters, rng=rng)[0]


def score(r):
    n = r["wins"] + r["draws"] + r["losses"]
    return (r["wins"] + 0.5 * r["draws"]) / n


def ladder(game, budgets, games=10):
    rows, prev = [], None
    for b in budgets:
        vs_rand = score(match(game, uct_player(game, b, 1), random_player(game, 2), games))
        vs_prev = None if prev is None else score(
            match(game, uct_player(game, b, 3), uct_player(game, prev, 4), games))
        rows.append((b, vs_rand, vs_prev))
        prev = b
    return rows


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    R, C, K = (int(x) for x in sys.argv[2:5]) if len(sys.argv) >= 5 else (6, 7, 4)
    g = ConnectN(R, C, K, gravity=True)
    print(f"board {R}x{C} k={K} games/cell={games}")
    print("iters  vs_random  vs_previous_budget")
    for b, a, p in ladder(g, [10, 40, 160, 640], games):
        print(f"{b:5d}  {a:9.2f}  {'-' if p is None else format(p, '.2f'):>18}")
