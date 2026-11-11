import json

from codesearch.__main__ import main
from codesearch.complexity import source_complexity

CASES = [
    ("def f():\n    return 1\n", 1),
    ("def f(x):\n    if x:\n        return 1\n    return 2\n", 2),
    ("def f(x):\n    if x and x > 1 or x < -5:\n        return 1\n", 4),
    ("def f(xs):\n    for x in xs:\n        while x:\n            x -= 1\n", 3),
    ("def f(xs):\n    return [x for x in xs if x if x > 1]\n", 4),
    ("def f():\n    try:\n        pass\n    except A:\n        pass\n    except B:\n        pass\n    else:\n        pass\n", 3),
    ("def f(x):\n    match x:\n        case 1:\n            return 1\n        case 2:\n            return 2\n        case _:\n            return 0\n", 3),
]


def test_scores():
    for src, want in CASES:
        assert source_complexity(src) == want, src


def test_indented_method_and_syntax_error():
    assert source_complexity("    def m(self, x):\n        return x if x else 0\n") == 2
    assert source_complexity("def broken(:\n") is None


def test_cli_ranks_and_filters(tmp_path, capsys):
    (tmp_path / "m.py").write_text(
        "def hairy(x):\n    if x:\n        for i in x:\n            if i and x:\n                return i\n    return 0\n\n\n"
        "def flat():\n    return 1\n")
    (tmp_path / "test_m.py").write_text("def test_a(x):\n    if x:\n        pass\n    if x:\n        pass\n")
    assert main(["complexity", "--root", str(tmp_path), "--no-store", "--min", "2", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [d["id"].split("::")[-1] for d in data] == ["hairy"]
    assert data[0]["complexity"] == 5
