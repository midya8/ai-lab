import random
from mctslab import TicTacToe
from mctslab.accuracy import reachable_states, optimal_move_rate
from mctslab.arena import perfect_player, random_player
from mctslab.mcts import uct_search


def test_reachable_count_tictactoe():
    # known: 5478 reachable positions incl. terminal; 4520 non-terminal
    assert len(reachable_states(TicTacToe())) == 4520


def test_perfect_player_rate_one():
    g = TicTacToe()
    st = reachable_states(g, limit=100)
    rate, bad = optimal_move_rate(g, perfect_player(g), st)
    assert rate == 1.0 and bad == []


def test_random_worse_and_sampling_deterministic():
    g = TicTacToe()
    assert reachable_states(g, 50, seed=1) == reachable_states(g, 50, seed=1)
    st = reachable_states(g, limit=200)
    rate, bad = optimal_move_rate(g, random_player(g), st)
    assert 0.2 < rate < 0.95 and len(bad) > 0


def test_more_iterations_not_worse():
    g = TicTacToe()
    st = reachable_states(g, limit=60)
    def mk(n):
        return lambda s: uct_search(g, s, iterations=n, rng=random.Random(0))[0]
    lo, _ = optimal_move_rate(g, mk(5), st)
    hi, _ = optimal_move_rate(g, mk(300), st)
    assert hi > lo and hi > 0.6
