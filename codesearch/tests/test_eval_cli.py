import os
import json

import pytest

from codesearch.__main__ import main
from codesearch.evaluate import LabelError, check_labels, dcg, format_table, load_queries, run_eval, score_mode
from codesearch.index import Index
from codesearch.search import Searcher
from codesearch.store import sync


@pytest.fixture
def queries(repo):
    qs = {"root": ".", "queries": [
        {"query": "retry with exponential backoff", "relevant": ["app/retry.py::retry_with_backoff"]},
        {"query": "turn a title into a url slug", "relevant": ["app/textutil.py::slugify"]},
        {"query": "atomic json write", "relevant": ["app/storage.py::save_json_atomically", "app/storage.py::load_json"]},
    ]}
    p = repo / "queries.json"
    p.write_text(json.dumps(qs))
    return p


def test_dcg_matches_formula():
    assert dcg([1, 0, 1]) == pytest.approx(1 + 0.5)


def test_scores_are_bounded_and_sensible(repo, queries):
    meta, cases = load_queries(queries)
    ix = Index.build(sync(repo, None)[0])
    scores = {s.mode: s for s in run_eval(ix, cases)}
    for s in scores.values():
        assert 0 <= s.recall_at_1 <= s.recall_at_5 <= s.recall_at_10 <= 1
        assert 0 <= s.mrr <= 1 and 0 <= s.ndcg_at_10 <= 1
    assert scores["bm25f"].mrr == 1.0 and scores["bm25f"].first_rank == [1, 1, 1]
    assert scores["bm25f"].mrr >= scores["name"].mrr


def test_stale_labels_fail_loudly(repo, queries):
    ix = Index.build(sync(repo, None)[0])
    bad = load_queries(queries)[1]
    bad[0].relevant = ["app/retry.py::gone"]
    with pytest.raises(LabelError, match="gone"):
        check_labels(ix, bad)


def test_empty_label_list_rejected(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"queries": [{"query": "x", "relevant": []}]}))
    with pytest.raises(LabelError):
        load_queries(p)


def test_rank_beyond_depth_counts_as_miss(repo, queries):
    ix = Index.build(sync(repo, None)[0])
    cases = load_queries(queries)[1]
    cases[0].relevant = ["app/textutil.py::word_count"]  # unrelated to the query
    sc = score_mode(Searcher(ix), cases[:1], "bm25f", depth=1)
    assert sc.recall_at_10 == 0.0 and sc.mrr == 0.0


def test_table_formats_all_modes(repo, queries):
    ix = Index.build(sync(repo, None)[0])
    t = format_table(run_eval(ix, load_queries(queries)[1]))
    assert t.count("\n") == 6 and "bm25f" in t and "nDCG@10" in t


def test_cli_search_json_and_exit_codes(repo, capsys):
    assert main(["search", "--root", str(repo), "--no-store", "--json", "url", "slug"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["qualname"] == "slugify" and out[0]["rank"] == 1
    assert main(["search", "--root", str(repo), "--no-store", "zzzzqqqq"]) == 1


def test_cli_index_then_search_uses_store(repo, capsys):
    assert main(["index", "--root", str(repo)]) == 0
    assert "4 added" in capsys.readouterr().out
    assert (repo / ".codesearch" / "index.json").exists()
    main(["index", "--root", str(repo)])
    assert "4 reused" in capsys.readouterr().out


def test_cli_show_marks_matching_lines(repo, capsys):
    main(["search", "--root", str(repo), "--no-store", "--show", "-k", "1", "json", "atomically"])
    out = capsys.readouterr().out
    assert "save_json_atomically" in out and ">" in out


def test_cli_similar_and_errors(repo, capsys):
    assert main(["similar", "--root", str(repo), "--no-store", "slugify"]) == 0
    assert main(["similar", "--root", str(repo), "--no-store", "nonexistent"]) == 2


def test_cli_eval_and_stats(repo, queries, capsys):
    assert main(["eval", str(queries), "--modes", "bm25f,name", "--misses"]) == 0
    out = capsys.readouterr().out
    assert "bm25f" in out and "| name" in out
    assert main(["stats", "--root", str(repo), "--no-store"]) == 0
    assert "chunks in 4 files" in capsys.readouterr().out


def test_cli_eval_reports_stale_labels(repo, tmp_path, capsys):
    p = repo / "q.json"
    p.write_text(json.dumps({"root": ".", "queries": [{"query": "x y", "relevant": ["nope.py::f"]}]}))
    assert main(["eval", str(p)]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_dups_runs(repo, capsys):
    assert main(["dups", "--root", str(repo), "--no-store"]) == 0
    assert "candidate pair" in capsys.readouterr().out


def test_shipped_query_files_are_well_formed():
    from pathlib import Path
    for name in ("queries_dev.json", "queries_heldout.json"):
        meta, cases = load_queries(Path(__file__).resolve().parent.parent / "bench" / name)
        assert len(cases) >= 20 and len({c.query for c in cases}) == len(cases)
        assert meta["include"] and all(c.relevant for c in cases)


@pytest.mark.skipif(not os.environ.get("CODESEARCH_MONOREPO"), reason="needs the full ai-lab checkout; set CODESEARCH_MONOREPO=1")
def test_shipped_labels_resolve_in_the_monorepo():
    from pathlib import Path
    bench = Path(__file__).resolve().parent.parent / "bench"
    for name in ("queries_dev.json", "queries_heldout.json"):
        meta, cases = load_queries(bench / name)
        root = (bench / meta["root"]).resolve()
        check_labels(Index.build(sync(root, None, meta["include"])[0]), cases)
