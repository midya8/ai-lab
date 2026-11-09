import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import hybrid_eval  # noqa: E402


def test_eval_runs_and_is_deterministic():
    a = hybrid_eval.run(seed=1, n=300, nq=40)
    b = hybrid_eval.run(seed=1, n=300, nq=40)
    assert a == b
    assert set(a) == {"dense", "bm25", "rrf", "weighted a=0.5", "weighted a=0.2"}


def test_bm25_solves_lookup_and_fusion_keeps_it():
    r = hybrid_eval.run(seed=2, n=300, nq=60)
    assert r["bm25"]["recall"]["lookup"] == 1.0
    assert r["rrf"]["recall"]["lookup"] >= 0.9
    assert r["dense"]["recall"]["lookup"] < r["bm25"]["recall"]["lookup"]
