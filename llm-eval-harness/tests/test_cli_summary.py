import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evalharness.cli import main

SUITE = os.path.join(os.path.dirname(__file__), "..", "suites", "basic.json")


def test_summary_file_written_and_regression_exit():
    d = tempfile.mkdtemp()
    base, summ = os.path.join(d, "b.json"), os.path.join(d, "s.md")
    assert main([SUITE, "--backend", "rules", "--save", base, "--summary-file", summ]) == 0
    assert os.path.getsize(summ) > 0
    assert main([SUITE, "--backend", "echo", "--baseline", base, "--summary-file", summ]) == 1
    with open(summ) as f:
        assert f.read().count("\n") > 2
