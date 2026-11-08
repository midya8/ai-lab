import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import rope_vs_learned as ex  # noqa: E402


def test_experiment_table_marks_learned_positions_na_past_block():
    args = type("A", (), dict(steps=3, block=16, layers=1, heads=2, embd=16, batch=4,
                              lr=3e-3, seed=0, lengths=[8, 16, 32]))()
    rows = ex.run(args)
    assert rows["learned"]["loss"][32] is None
    assert rows["rope"]["loss"][32] is not None
    assert rows["rope"]["params"] < rows["learned"]["params"]
    table = ex.fmt(rows, args.lengths)
    assert "n/a" in table and "| rope |" in table
