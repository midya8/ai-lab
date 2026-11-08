import os, sys, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evalharness import EchoBackend, RuleBasedBackend, load_suite, run_matrix, matrix_table, render_matrix_markdown
from evalharness.cli import main

SUITE = os.path.join(os.path.dirname(__file__), "..", "suites", "basic.json")


def test_matrix_ranks_rules_over_echo():
    m = run_matrix([EchoBackend(), RuleBasedBackend()], load_suite(SUITE))
    t = matrix_table(m)
    assert t["backends"] == ["echo", "rules"]
    assert t["winners"] == ["rules"] and t["means"][1] > t["means"][0]
    assert len(t["rows"]) == len(load_suite(SUITE)["cases"])
    assert t["contested"]


def test_tie_lists_both_winners():
    t = matrix_table({"suite": "s", "runs": [
        {"backend": "a", "mean_score": 1.0, "results": [{"id": "x", "score": 1.0}]},
        {"backend": "b", "mean_score": 1.0, "results": [{"id": "x", "score": 1.0}]}]})
    assert t["winners"] == ["a", "b"] and t["contested"] == []


def test_duplicate_backend_names_rejected():
    with pytest.raises(ValueError):
        run_matrix([EchoBackend(), EchoBackend()], load_suite(SUITE))


def test_markdown_and_cli(capsys, tmp_path):
    f = tmp_path / "s.md"
    assert main([SUITE, "--matrix", "echo,rules", "--summary-file", str(f)]) == 0
    out = capsys.readouterr().out
    assert "| case | echo | rules |" in out and "Best: rules" in out
    assert f.read_text().startswith("# Matrix")
