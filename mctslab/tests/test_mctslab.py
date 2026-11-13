import random, pytest
from mctslab import TicTacToe, ConnectN, MCTS, uct_search, minimax_value, best_moves
from mctslab.arena import match, random_player, perfect_player, play_game


def test_ttt_value_is_draw():
    g = TicTacToe()
    assert minimax_value(g, g.initial()) == 0


def test_winner_detection_and_illegal():
    g = TicTacToe()
    s = g.initial()
    for m in (0, 3, 1, 4, 2):
        s = g.play(s, m)
    assert g.winner(s) == 1 and g.legal_moves(s) == []
    with pytest.raises(ValueError):
        g.play(g.initial(), 9)


def test_gravity_drops():
    g = ConnectN(4, 4, 3, gravity=True)
    s = g.play(g.initial(), 2)
    assert s[0][12 + 2] == 1
    s = g.play(s, 2)
    assert s[0][8 + 2] == -1


def test_mcts_takes_immediate_win_and_blocks():
    g = TicTacToe()
    s = g.initial()
    for m in (0, 3, 1, 4):  # X to move, X wins at 2
        s = g.play(s, m)
    assert uct_search(g, s, 300)[0] == 2
    s = g.initial()
    for m in (0, 4, 1):  # O to move must block at 2
        s = g.play(s, m)
    assert uct_search(g, s, 800)[0] == 2


def test_mcts_never_loses_to_random():
    g = TicTacToe()
    r = match(g, MCTS(g, 400, seed=1), random_player(g, 1), games=10)
    assert r["losses"] == 0 and r["wins"] >= 7


def test_mcts_draws_perfect_play():
    g = TicTacToe()
    r = match(g, MCTS(g, 1500, seed=2), perfect_player(g, 2), games=4)
    assert r["losses"] == 0


def test_more_iterations_agree_more_with_minimax():
    g = TicTacToe()
    rng = random.Random(5)
    states = []
    for _ in range(25):
        s = g.initial()
        for _ in range(rng.randint(1, 4)):
            s = g.play(s, rng.choice(g.legal_moves(s)))
        if not g.is_terminal(s):
            states.append(s)
    def agree(it):
        return sum(uct_search(g, s, it, rng=random.Random(0))[0] in best_moves(g, s) for s in states)
    assert agree(1000) >= agree(20)
    assert agree(1000) >= int(0.8 * len(states))
