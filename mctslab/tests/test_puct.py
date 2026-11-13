import random
from mctslab import TicTacToe, ConnectN, PUCT, puct_search, minimax_value, best_moves, MCTS, line_heuristic_value
from mctslab.arena import match, random_player


def _state(g, moves):
    s = g.initial()
    for m in moves:
        s = g.play(s, m)
    return s


def test_takes_immediate_win():
    g = TicTacToe()
    s = _state(g, [0, 3, 1, 4])  # X to move, 2 wins
    assert puct_search(g, s, 200)[0] == 2


def test_blocks_threat():
    g = TicTacToe()
    s = _state(g, [0, 4, 1])  # O to move must block at 2
    assert puct_search(g, s, 400)[0] == 2


def test_visit_counts_sum_and_prior_steers():
    g = TicTacToe()
    s = g.initial()
    pri = lambda gm, st: {m: (1.0 if m == 4 else 0.0) for m in gm.legal_moves(st)}
    _, counts = puct_search(g, s, 50, prior_fn=pri, value_fn=lambda gm, st: 0.0)
    assert sum(counts.values()) == 49 or sum(counts.values()) == 50
    assert max(counts, key=counts.get) == 4


def test_heuristic_value_sign_and_terminal():
    g = ConnectN(4, 4, 3, gravity=True)
    s = _state(g, [0, 1, 0])  # X has two stacked in col 0
    assert line_heuristic_value(g, s) > 0
    w = _state(TicTacToe(), [0, 3, 1, 4, 2])
    assert line_heuristic_value(TicTacToe(), w) == 1.0


def test_never_loses_to_random_and_deterministic():
    g = TicTacToe()
    r = match(g, PUCT(g, 300, seed=1), random_player(g, 2), games=10)
    assert r["losses"] == 0
    a = puct_search(g, g.initial(), 100, rng=random.Random(5))
    b = puct_search(g, g.initial(), 100, rng=random.Random(5))
    assert a == b
