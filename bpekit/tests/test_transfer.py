import pytest
from bpekit.transfer import transfer_matrix, transfer_penalty, format_matrix

PROSE = "the quick brown fox jumps over the lazy dog and then rests. " * 60
CODE = "def add(a, b):\n    return a + b\n\nfor i in range(10):\n    print(add(i, 1))\n" * 40


def test_diagonal_beats_off_domain():
    names, m = transfer_matrix({"prose": PROSE, "code": CODE}, vocab_size=330)
    for j in range(2):
        assert m[j][j] >= m[1 - j][j]


def test_penalty_nonnegative_and_table():
    names, m = transfer_matrix({"prose": PROSE, "code": CODE}, vocab_size=330)
    p = transfer_penalty(names, m)
    assert set(p) == {"prose", "code"} and all(v >= 0 for v in p.values())
    t = format_matrix(names, m)
    assert t.count("\n") == 3 and "prose" in t


def test_validation():
    with pytest.raises(ValueError):
        transfer_matrix({"a": PROSE})
    with pytest.raises(ValueError):
        transfer_matrix({"a": PROSE, "b": CODE}, holdout=1.5)
