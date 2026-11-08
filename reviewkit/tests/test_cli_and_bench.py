import json
import os
import subprocess
import sys

import pytest

from reviewkit.bench import expected_for, load_cases, make_diff, run_bench
from reviewkit.cli import main

HERE = os.path.dirname(__file__)
BENCH = os.path.join(HERE, "..", "bench")
RISKY = "def f(x):\n    return eval(x)\n"


def test_dev_bench_is_perfect_because_rules_were_tuned_on_it():
    r = run_bench(load_cases(os.path.join(BENCH, "dev_cases.py")))
    assert r.cases == 20 and r.precision() == 1.0 and r.recall() == 1.0


def test_heldout_bench_floor():
    r = run_bench(load_cases(os.path.join(BENCH, "heldout_cases.py")))
    # Documented known misses: the SQL-with-concatenated-placeholders and moved-code false positives.
    assert r.precision() >= 0.9 and r.recall() >= 0.95
    assert len(r.false_positives) == 2


def test_expect_marker_must_be_on_a_new_line():
    with pytest.raises(AssertionError):
        expected_for({"a.py": ("x = 1  # EXPECT: py-eval\n", "x = 1  # EXPECT: py-eval\ny = 2\n")})


def test_cli_review_fail_on_and_formats(tmp_path, capsys):
    (tmp_path / "a.py").write_text(RISKY)
    d = tmp_path / "p.diff"
    d.write_text(make_diff({"a.py": (None, RISKY)}))
    base = ["review", "--diff", str(d), "--root", str(tmp_path)]
    assert main(base) == 0
    assert "a.py:2: [high] py-eval" in capsys.readouterr().out
    assert main(base + ["--fail-on", "high"]) == 1
    assert main(base + ["--fail-on", "critical"]) == 0
    capsys.readouterr()
    assert main(base + ["--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["findings"][0]["rule"] == "py-eval" and len(data["findings"][0]["fp"]) == 12
    assert main(base + ["--format", "github"]) == 0
    assert json.loads(capsys.readouterr().out)["comments"][0]["line"] == 2


def test_cli_reads_git_diff(tmp_path, capsys):
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text("x = 1\n")
    run("add", "-A"); run("commit", "-qm", "base")
    run("checkout", "-qb", "feat")
    (tmp_path / "a.py").write_text("x = 1\n" + RISKY)
    run("add", "-A"); run("commit", "-qm", "feat")
    assert main(["review", "--base", "main", "--root", str(tmp_path), "--config", str(tmp_path / "none.json")]) == 0
    assert "a.py:3: [high] py-eval" in capsys.readouterr().out


def test_cli_requires_a_diff_source_and_valid_provider():
    with pytest.raises(SystemExit):
        main(["review"])
    with pytest.raises(SystemExit):
        main(["review", "--diff", "-", "--provider", "openai"])


def test_cli_post_needs_token(tmp_path, monkeypatch):
    d = tmp_path / "p.diff"
    d.write_text(make_diff({"a.py": (None, "x = 1\n")}))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        main(["review", "--diff", str(d), "--post", "o/r#1"])


def test_rules_command_lists_every_rule(capsys):
    from reviewkit.rules import RULE_TABLE
    main(["rules"])
    out = capsys.readouterr().out
    assert all(r in out for r in RULE_TABLE)
