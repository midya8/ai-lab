from bpekit.bpe import train_fast
from bpekit.usage import usage

TEXT = "the cat sat on the mat. " * 50 + "zebra quartz " * 5


def test_counts_match_encoding():
    tok = train_fast(TEXT, 290)
    u = usage(tok, TEXT)
    assert u["tokens"] == len(tok.encode(TEXT))
    assert 0 <= u["unused"] <= len(tok.merges)


def test_unseen_text_leaves_tokens_unused():
    tok = train_fast(TEXT, 290)
    u = usage(tok, "zzzz")
    assert u["unused_frac"] > 0.8
    assert u["top"] == [] or all(len(b) >= 2 for b, _ in u["top"])


def test_top_sorted_and_entropy_positive():
    tok = train_fast(TEXT, 290)
    u = usage(tok, TEXT, top=5)
    counts = [v for _, v in u["top"]]
    assert counts == sorted(counts, reverse=True)
    assert u["entropy_bits"] > 0
