import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import context_extension as ex  # noqa: E402


def test_experiment_table_has_every_length_and_method():
    args = type("A", (), dict(steps=3, block=16, layers=1, heads=2, embd=16, batch=4, lr=3e-3, seeds=[0],
                              lengths=[32], ft_steps=2, ft_batch=2, ft_lr=1e-3, stride=8))()
    res = ex.run(args)
    assert set(res["cells"]) == {(32, m) for m in ex.MODES}
    assert all(0 < v < 10 for cell in res["cells"].values() for v in cell)
    table = ex.fmt(res, args.lengths)
    for m in ex.MODES:
        assert f"| 32 | {m} |" in table
