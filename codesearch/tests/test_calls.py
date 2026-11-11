import pytest

from codesearch.index import Index
from codesearch.search import Searcher
from codesearch.store import sync

SRC = '''
def helper(x):
    return x + 1


def middle(x):
    return helper(x) * 2


def top(x):
    return middle(x) + helper(x)


class Box:
    def run(self):
        return top(1)
'''


@pytest.fixture
def s(tmp_path):
    (tmp_path / "m.py").write_text(SRC)
    (tmp_path / "test_m.py").write_text("from m import top\n\ndef test_top():\n    assert top(1)\n")
    chunks, _ = sync(tmp_path, None)
    return Searcher(Index.build(chunks))


def names(cs):
    return sorted(c.qualname for c in cs)


def test_callees(s):
    assert names(s.calls("top")["callees"]) == ["helper", "middle"]


def test_callers_exclude_tests_by_default(s):
    assert "Box.run" in names(s.calls("top")["callers"])
    assert "test_top" not in names(s.calls("top")["callers"])


def test_callers_include_tests_on_request(s):
    assert "test_top" in names(s.calls("top", include_tests=True)["callers"])


def test_leaf_has_no_callees(s):
    assert s.calls("helper")["callees"] == []
    assert names(s.calls("helper")["callers"]).count("middle") == 1


def test_unknown_raises(s):
    with pytest.raises(KeyError):
        s.calls("nope")
