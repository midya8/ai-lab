import random
import pytest
from bpekit.sweep import sweep, format_table


def _text():
    r = random.Random(1)
    w = ["the", "tokenizer", "merges", "bytes", "pairs", "vocab", "model", "token"]
    return " ".join(r.choice(w) for _ in range(4000))


def test_monotone_train_compression():
    rows = sweep(_text(), [260, 280, 300])
    t = [r[2] for r in rows]
    assert t == sorted(t) and t[-1] > t[0]


def test_sorted_dedup_and_table():
    rows = sweep(_text(), [300, 260, 260])
    assert [r[0] for r in rows] == [260, 300]
    assert "heldout_bpt" in format_table(rows)


def test_bad_args():
    with pytest.raises(ValueError):
        sweep("abc", [260], holdout=1.5)
    with pytest.raises(ValueError):
        sweep("", [260])
