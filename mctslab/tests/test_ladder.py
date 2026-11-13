import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
from mctslab import ConnectN
from strength_ladder import ladder, score


def test_score_counts_draws_half():
    assert score({"wins": 1, "draws": 2, "losses": 1}) == 0.5


def test_ladder_small_board_shape_and_range():
    g = ConnectN(4, 4, 3, gravity=True)
    rows = ladder(g, [5, 60], games=4)
    assert [r[0] for r in rows] == [5, 60]
    assert rows[0][2] is None and 0 <= rows[1][2] <= 1
    assert all(0 <= r[1] <= 1 for r in rows)
    assert rows[1][1] >= 0.75  # 60 iterations should beat random on 4x4/k=3
