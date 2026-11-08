import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ragevals.sweep import sweep, best, to_markdown
from ragevals.cli import main

D = pathlib.Path(__file__).resolve().parents[1] / "data"
DATA = json.loads((D / "hard_eval.json").read_text())


def test_smaller_chunks_make_more_chunks():
    rows = sweep(str(D / "hard_docs"), DATA, [15, 100], ["bm25"])
    by = {r["max_words"]: r for r in rows}
    assert by[15]["chunks"] > by[100]["chunks"]


def test_rows_cover_grid_and_metrics_bounded():
    rows = sweep(str(D / "hard_docs"), DATA, [20, 60], ["bm25", "lsa"])
    assert len(rows) == 4
    assert all(0 <= r["mrr"] <= 1 and 0 <= r["recall@3"] <= 1 for r in rows)


def test_best_prefers_larger_on_tie():
    rows = [{"max_words": 20, "mrr": .5}, {"max_words": 60, "mrr": .5}, {"max_words": 10, "mrr": .4}]
    assert best(rows)["max_words"] == 60


def test_markdown_and_cli(capsys):
    rows = sweep(str(D / "hard_docs"), DATA, [30], ["bm25"])
    assert to_markdown(rows).count("\n") == 2
    assert main(["--docs", str(D / "hard_docs"), "--dataset", str(D / "hard_eval.json"), "--sweep", "30,60", "--method", "all"]) == 0
    out = capsys.readouterr().out
    assert "| 30 | hybrid |" in out and "| 60 | lsa |" in out
