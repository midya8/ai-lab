import pytest
from mctslab import TicTacToe
from mctslab.root_parallel import root_parallel_search, RootParallelMCTS
from mctslab.arena import match, random_player


def _play(g, moves):
    s = g.initial()
    for m in moves:
        s = g.play(s, m)
    return s


def test_budget_split_and_merge():
    g = TicTacToe()
    _, counts = root_parallel_search(g, g.initial(), iterations=403, workers=4)
    assert sum(counts.values()) == 403


def test_takes_win():
    g = TicTacToe()
    assert root_parallel_search(g, _play(g, [0, 3, 1, 4]), 400, 4)[0] == 2


def test_deterministic_and_process_equivalent():
    g = TicTacToe()
    s = _play(g, [0, 4])
    a = root_parallel_search(g, s, 200, 2, seed=3)
    b = root_parallel_search(g, s, 200, 2, seed=3, processes=True)
    assert a == b


def test_bad_workers():
    g = TicTacToe()
    with pytest.raises(ValueError):
        root_parallel_search(g, g.initial(), 10, 0)


def test_beats_random():
    g = TicTacToe()
    r = match(g, RootParallelMCTS(g, 200, 4), random_player(g, 1), games=20)
    assert r["wins"] >= 14
