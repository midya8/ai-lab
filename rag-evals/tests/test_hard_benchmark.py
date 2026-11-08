import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ragevals.cli import build_index, METHODS
from ragevals.metrics import evaluate

ROOT = pathlib.Path(__file__).resolve().parents[1] / "data"
DATA = json.loads((ROOT / "hard_eval.json").read_text())


def run(method, k=3):
    idx = build_index(str(ROOT / "hard_docs"), method=method)
    return evaluate(lambda q, k: [c.doc_id for c, _ in idx.search(q, k)], DATA, k)


def test_labels_reference_existing_docs():
    names = {p.stem for p in (ROOT / "hard_docs").glob("*.txt")}
    for item in DATA:
        assert item["relevant_docs"] and set(item["relevant_docs"]) <= names


def test_benchmark_is_not_saturated():
    for m in METHODS:
        r = run(m)
        assert r["recall@3"] < 0.95, f"{m} saturated the hard set"
        assert r["recall@3"] > 0.5, f"{m} regressed badly"


def test_recall_monotone_in_k():
    assert run("bm25", 5)["recall@5"] >= run("bm25", 1)["recall@1"]
