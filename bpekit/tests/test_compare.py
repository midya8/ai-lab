from bpekit import train_fast
from bpekit.compare import compare

TXT = "the cat sat on the mat. the dog ate the bone. " * 30


def test_identical_tokenizers():
    t = train_fast(TXT, 300)
    r = compare(t, t, TXT)
    assert r["ratio"] == 1.0 and r["boundary_jaccard"] == 1.0


def test_bigger_vocab_compresses_better():
    small, big = train_fast(TXT, 270), train_fast(TXT, 400)
    r = compare(big, small, TXT)
    assert r["tokens_a"] < r["tokens_b"] and r["ratio"] < 1
    assert r["bpt_a"] > r["bpt_b"]
    assert 0 < r["boundary_jaccard"] < 1


def test_unicode_and_empty():
    t = train_fast(TXT, 300)
    assert compare(t, t, "héllo ✓")["boundary_jaccard"] == 1.0
    assert compare(t, t, "")["tokens_a"] == 0
