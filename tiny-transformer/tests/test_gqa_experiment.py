import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gqa_vs_mha as ex  # noqa: E402


def test_experiment_reports_shrinking_params_and_cache():
    args = type("A", (), dict(steps=3, block=16, layers=1, heads=4, embd=16, batch=4,
                              lr=3e-3, seed=0, kv_heads=[4, 2, 1]))()
    rows = ex.run(args)
    assert [r["kv_heads"] for r in rows] == [4, 2, 1]
    assert rows[0]["params"] > rows[1]["params"] > rows[2]["params"]
    assert rows[0]["cache_bytes"] == 2 * rows[1]["cache_bytes"] == 4 * rows[2]["cache_bytes"]
    table = ex.fmt(rows, 4)
    assert "| MHA |" in table and "| GQA |" in table and "| MQA |" in table and "0.25x" in table
