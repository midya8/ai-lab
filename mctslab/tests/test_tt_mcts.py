import random
from mctslab import TicTacToe, ConnectN, minimax_value
from mctslab.tt_mcts import tt_uct_search, TTMCTS
from mctslab.arena import match, random_player


def _play(g, moves):
    s = g.initial()
    for m in moves:
        s = g.play(s, m)
    return s


def test_takes_win_and_blocks():
    g = TicTacToe()
    assert tt_uct_search(g, _play(g, [0, 3, 1, 4]), 400)[0] == 2
    s = _play(g, [0, 4, 1])  # O to move must block at 2
    assert tt_uct_search(g, s, 800)[0] == 2


def test_root_visits_sum_to_iterations():
    g = TicTacToe()
    _, counts = tt_uct_search(g, g.initial(), 300)
    assert sum(counts.values()) == 300


def test_table_shares_transpositions():
    g = TicTacToe()
    _, _, table = tt_uct_search(g, g.initial(), 3000, return_table=True)
    # tic-tac-toe has at most 5478 reachable states; sharing keeps the table far below iterations*depth
    assert len(table) <= 5478
    s_a = _play(g, [0, 4, 8])
    s_b = _play(g, [8, 4, 0])
    assert s_a == s_b and s_a in table


def test_deterministic_with_seed():
    g = TicTacToe()
    a = tt_uct_search(g, g.initial(), 200, rng=random.Random(5))
    b = tt_uct_search(g, g.initial(), 200, rng=random.Random(5))
    assert a == b


def test_does_not_lose_to_random():
    g = TicTacToe()
    r = match(g, TTMCTS(g, 300), random_player(g), games=6)
    assert r["losses"] == 0
