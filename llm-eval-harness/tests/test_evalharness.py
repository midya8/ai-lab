import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evalharness import *  # noqa
from evalharness.cli import main
from evalharness.scorers import score

SUITE = os.path.join(os.path.dirname(__file__), "..", "suites", "basic.json")


def test_scorers():
    assert score("exact", "Paris!", {"expected": "paris"}) == 1
    assert score("numeric", "so 3.0", {"expected": 3}) == 1
    assert score("json_keys", '{"a":1}', {"expected": ["a", "b"]}) == 0
    assert 0 < score("token_f1", "the cat sat", {"expected": "the cat"}) < 1
    assert score("regex", "id-42", {"expected": r"\d+"}) == 1


def test_run_rules_backend():
    run = run_suite(get_backend("rules"), load_suite(SUITE))
    assert run["n"] == 7 and run["errors"] == 0
    assert abs(run["mean_score"] - 6 / 7) < 1e-9
    assert run["by_tag"]["math"] == 1.0


class Boom(Backend):
    name = "boom"

    def generate(self, p):
        raise RuntimeError("down")


def test_backend_errors_are_scored_zero():
    run = run_suite(Boom(), load_suite(SUITE))
    assert run["errors"] == 7 and run["mean_score"] == 0


def test_regression_detection(tmp_path):
    good = run_suite(get_backend("rules"), load_suite(SUITE))
    bad = run_suite(get_backend("echo"), load_suite(SUITE))
    d = compare(good, bad)
    assert not d["passed"] and d["regressions"]
    assert compare(good, good)["passed"]
    assert "REGRESSION" in render_markdown(bad, d)


def test_cli_baseline_exit_codes(tmp_path):
    base = str(tmp_path / "b.json")
    assert main([SUITE, "--backend", "rules", "--save", base]) == 0
    assert main([SUITE, "--backend", "rules", "--baseline", base]) == 0
    assert main([SUITE, "--backend", "echo", "--baseline", base]) == 1
