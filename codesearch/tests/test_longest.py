import json

from codesearch.__main__ import main


def fn(name, n):
    return f"def {name}(x):\n" + "".join(f"    x += {i}\n" for i in range(n - 2)) + "    return x\n"


def test_ranked_longest_first(tmp_path, capsys):
    (tmp_path / "m.py").write_text(fn("short", 5) + "\n\n" + fn("big", 50) + "\n\n" + fn("mid", 45))
    assert main(["longest", "--root", str(tmp_path), "--no-store", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [(d["id"].split("::")[-1], d["lines"]) for d in data] == [("big", 50), ("mid", 45)]


def test_min_and_tests_excluded(tmp_path, capsys):
    (tmp_path / "m.py").write_text(fn("f", 10))
    (tmp_path / "test_m.py").write_text(fn("test_f", 60))
    assert main(["longest", "--root", str(tmp_path), "--no-store", "--json", "--min", "10"]) == 0
    assert [d["id"].split("::")[-1] for d in json.loads(capsys.readouterr().out)] == ["f"]
    assert main(["longest", "--root", str(tmp_path), "--no-store", "--json", "--min", "50", "--tests"]) == 0
    assert [d["id"].split("::")[-1] for d in json.loads(capsys.readouterr().out)] == ["test_f"]


def test_text_output(tmp_path, capsys):
    (tmp_path / "m.py").write_text(fn("f", 12))
    assert main(["longest", "--root", str(tmp_path), "--no-store", "--min", "10"]) == 0
    out = capsys.readouterr().out
    assert "m.py:1  f [function]" in out and "1 function(s) of at least 10 lines" in out
