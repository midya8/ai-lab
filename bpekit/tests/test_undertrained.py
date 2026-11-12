from bpekit.bpe import train_fast
from bpekit.undertrained import undertrained

TEXT = "the cat sat on the mat. " * 50 + "zebra quartz " * 5


def test_never_used_tokens_found():
    tok = train_fast(TEXT, 290)
    r = undertrained(tok, "zzzz")
    assert r and all(n == 0 for _, _, n in r)
    assert all(i >= 256 for i, _, _ in r)


def test_threshold_is_monotone_and_sorted():
    tok = train_fast(TEXT, 290)
    a = undertrained(tok, TEXT, 0)
    b = undertrained(tok, TEXT, 10)
    assert {x[0] for x in a} <= {x[0] for x in b}
    counts = [n for _, _, n in b]
    assert counts == sorted(counts)


def test_limit():
    tok = train_fast(TEXT, 290)
    assert len(undertrained(tok, "zzzz", 0, limit=3)) == 3
