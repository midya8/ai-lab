import json

from codesearch.__main__ import main
from codesearch.index import Index
from codesearch.search import Searcher
from codesearch.store import sync

SRC = '''
def used_a_lot(x):
    y = x + 1
    return y


def rarely(x):
    y = x + 2
    return y


def documented(x):
    """Has docs."""
    y = x + 3
    return used_a_lot(y)


def _private(x):
    y = x
    return y


def tiny(): return 1


def caller_one():
    a = used_a_lot(1)
    return a


def caller_two():
    a = used_a_lot(2) + rarely(1)
    return a
'''


def searcher(tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    (tmp_path / "test_m.py").write_text("def test_x():\n    a = 1\n    assert used_a_lot(a)\n")
    chunks, _ = sync(tmp_path, None)
    return Searcher(Index.build(chunks))


def test_ranked_by_references_and_filters(tmp_path):
    res = searcher(tmp_path).undocumented()
    names = [c.qualname for c, _ in res]
    assert names[0] == "used_a_lot"
    assert dict((c.qualname, n) for c, n in res)["used_a_lot"] == 3  # documented, caller_one, caller_two
    assert "documented" not in names and "_private" not in names and "tiny" not in names
    assert not any(c.is_test for c, _ in res)


def test_min_lines_and_tests_flag(tmp_path):
    s = searcher(tmp_path)
    assert "tiny" in [c.qualname for c, _ in s.undocumented(min_lines=1)]
    assert any(c.is_test for c, _ in s.undocumented(include_tests=True))


def test_cli_json(tmp_path, capsys):
    (tmp_path / "m.py").write_text(SRC)
    assert main(["undocumented", "--root", str(tmp_path), "--no-store", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["id"].endswith("::used_a_lot") and data[0]["references"] == 3
