import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evalharness.cli import main
from evalharness.report import render_html

SUITE = os.path.join(os.path.dirname(__file__), "..", "suites", "basic.json")


def _run(out="<script>x</script>", err=None, score=0.0):
    return {"suite": "s<b>", "backend": "b", "n": 1, "mean_score": score, "errors": 0,
            "by_tag": {"t": score}, "results": [{"id": "c1", "output": out, "score": score,
            "error": err, "latency_ms": 1.0, "tags": ["t"]}]}


def test_html_escapes_and_marks_regression():
    diff = {"delta": -1.0, "passed": False, "regressions": [("c1", 1.0, 0.0)], "improvements": []}
    h = render_html(_run(), diff)
    assert "<script>x" not in h and "&lt;script&gt;" in h and "s&lt;b&gt;" in h
    assert 'class="reg"' in h and "REGRESSION" in h


def test_cli_writes_html():
    path = os.path.join(tempfile.mkdtemp(), "r.html")
    assert main([SUITE, "--backend", "rules", "--html", path]) == 0
    assert open(path).read().startswith("<!doctype html>")
