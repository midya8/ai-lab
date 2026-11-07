import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
from ragevals import chunk_text, BM25Index, recall_at_k, mrr, ndcg_at_k, evaluate
from ragevals.cli import build_index, main

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_chunk_respects_limit_and_overlaps():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    cs = chunk_text("d", text, max_words=6, overlap_sents=1)
    assert len(cs) >= 3
    assert all(len(c.text.split()) <= 6 for c in cs)
    assert cs[0].text.split(". ")[-1].rstrip(".") in cs[1].text
    assert cs[0].id == "d#0"


def test_chunk_empty_and_invalid():
    assert chunk_text("d", "") == []
    with pytest.raises(ValueError):
        chunk_text("d", "x", max_words=0)


def test_bm25_ranks_relevant_first():
    idx = build_index(ROOT / "data" / "docs")
    assert idx.search("LoRA low-rank matrices", 1)[0][0].doc_id == "lora"
    assert idx.search("zzzz unknown", 3) == []


def test_metrics():
    assert recall_at_k(["a", "b"], {"b", "c"}, 2) == 0.5
    assert mrr(["x", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], {"a"}, 3) == 1.0
    assert ndcg_at_k(["x"], {"a"}, 3) == 0.0


def test_end_to_end_eval_quality(capsys):
    assert main(["--docs", str(ROOT / "data/docs"), "--dataset", str(ROOT / "data/eval.json"), "--k", "3"]) == 0
    assert '"mrr"' in capsys.readouterr().out
    idx = build_index(ROOT / "data" / "docs")
    import json
    data = json.loads((ROOT / "data/eval.json").read_text())
    res = evaluate(lambda q, k: [c.doc_id for c, _ in idx.search(q, k)], data, 3)
    assert res["recall@3"] >= 0.85 and res["mrr"] >= 0.7
