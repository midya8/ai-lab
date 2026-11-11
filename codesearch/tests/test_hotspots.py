import json

from codesearch.__main__ import main

HAIRY = "def {n}(x):\n    if x:\n        for i in x:\n            if i and x:\n                return i\n    return 0\n"


def test_risk_orders_by_references(tmp_path, capsys):
    (tmp_path / "m.py").write_text(
        HAIRY.format(n="popular") + "\n\n" + HAIRY.format(n="lonely") + "\n\n"
        "def a():\n    return popular([1])\n\n\ndef b():\n    return popular([2])\n")
    assert main(["hotspots", "--root", str(tmp_path), "--no-store", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    names = [d["id"].split("::")[-1] for d in data]
    assert names == ["popular", "lonely"]
    assert data[0]["references"] == 2 and data[0]["risk"] == 5 * 3
    assert data[1]["references"] == 0 and data[1]["risk"] == 5


def test_min_and_tests_excluded(tmp_path, capsys):
    (tmp_path / "m.py").write_text(HAIRY.format(n="f"))
    (tmp_path / "test_m.py").write_text(HAIRY.format(n="test_f"))
    assert main(["hotspots", "--root", str(tmp_path), "--no-store", "--json", "--min", "6"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert main(["hotspots", "--root", str(tmp_path), "--no-store", "--json"]) == 0
    assert [d["id"].split("::")[-1] for d in json.loads(capsys.readouterr().out)] == ["f"]


def test_text_output(tmp_path, capsys):
    (tmp_path / "m.py").write_text(HAIRY.format(n="f"))
    assert main(["hotspots", "--root", str(tmp_path), "--no-store"]) == 0
    assert "m.py:1  f [function]" in capsys.readouterr().out


def test_dunder_methods_not_inflated(tmp_path, capsys):
    (tmp_path / "m.py").write_text(
        "class A:\n    def __init__(self, x):\n        if x:\n            for i in x:\n                if i and x:\n                    pass\n\n\n"
        "def use():\n    return A([1]).__init__([2])\n")
    assert main(["hotspots", "--root", str(tmp_path), "--no-store", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["references"] == 0
