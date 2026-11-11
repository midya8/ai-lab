import json

from codesearch.__main__ import main


def _names(capsys):
    return [d["id"].split("::")[-1] for d in json.loads(capsys.readouterr().out)]


def _proj(tmp_path):
    (tmp_path / "a.py").write_text(
        "def used():\n    return 1\n\n\ndef orphan():\n    return 2\n\n\n"
        "def main():\n    return used()\n\n\n"
        "class K:\n    def __init__(self):\n        self.x = 1\n\n    def lonely(self):\n        return 3\n")
    (tmp_path / "test_a.py").write_text("from a import orphan\n\n\ndef test_o():\n    assert orphan() == 2\n")


def test_reports_only_unmentioned(tmp_path, capsys):
    _proj(tmp_path)
    assert main(["unreferenced", "--root", str(tmp_path), "--no-store", "--json"]) == 0
    got = _names(capsys)
    assert "orphan" in got and "K.lonely" in got
    assert "used" not in got and "main" not in got and "K.__init__" not in got
    assert "test_o" not in got


def test_tests_count_as_references_when_asked(tmp_path, capsys):
    _proj(tmp_path)
    assert main(["unreferenced", "--root", str(tmp_path), "--no-store", "--json", "--tests"]) == 0
    got = _names(capsys)
    assert "orphan" not in got and "K.lonely" in got


def test_ignore_and_self_recursion(tmp_path, capsys):
    (tmp_path / "r.py").write_text("def rec(n):\n    return rec(n - 1) if n else 0\n\n\ndef keep():\n    return 1\n")
    assert main(["unreferenced", "--root", str(tmp_path), "--no-store", "--json", "--ignore", "keep"]) == 0
    got = _names(capsys)
    assert got == ["rec"]
