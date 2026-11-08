import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loralab.cli import rank_sweep, format_sweep


def test_sweep_shape_and_param_scaling():
    rows = rank_sweep(ranks=(1, 2, 4), steps=150)
    assert [r["rank"] for r in rows] == [1, 2, 4, None]
    p = [r["trainable_params"] for r in rows[:3]]
    assert p[1] == 2 * p[0] and p[2] == 4 * p[0]
    assert all(r["trainable_params"] < rows[-1]["trainable_params"] for r in rows[:3])


def test_sweep_is_deterministic_and_rank_helps():
    a = rank_sweep(ranks=(1, 8), steps=150)
    assert a == rank_sweep(ranks=(1, 8), steps=150)
    assert a[1]["acc_task_B"] >= a[0]["acc_task_B"] - 0.02


def test_format_lists_every_row():
    txt = format_sweep(rank_sweep(ranks=(2,), steps=50))
    assert "full" in txt and len(txt.splitlines()) == 3
