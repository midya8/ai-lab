from mctslab import TicTacToe
from mctslab.accuracy import reachable_states
from mctslab.value_error import estimate_value, value_error_curve
from mctslab.minimax import minimax_value


def test_estimate_in_range_and_none_on_forced():
    g = TicTacToe()
    v = estimate_value(g, g.initial(), 200)
    assert -1.0 <= v <= 1.0
    s = g.initial()
    for m in (0, 3, 1, 4, 2):  # X wins -> terminal
        s = g.play(s, m)
    assert estimate_value(g, s, 50) is None


def test_clear_win_estimated_positive():
    g = TicTacToe()
    s = g.initial()
    for m in (0, 3, 1, 4):  # X to move, 2 wins immediately
        s = g.play(s, m)
    assert minimax_value(g, s) == 1
    assert estimate_value(g, s, 300) > 0.5


def test_curve_error_shrinks_with_budget():
    g = TicTacToe()
    sts = reachable_states(g, limit=60, seed=1)
    c = value_error_curve(g, sts, budgets=(10, 400))
    assert c[1]["mae"] < c[0]["mae"]
    assert c[0]["n"] == c[1]["n"] > 0
