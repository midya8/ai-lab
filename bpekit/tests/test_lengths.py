from bpekit.bpe import train_fast
from bpekit.lengths import length_hist

TEXT = "the cat sat on the mat. " * 50


def test_hist_sums_to_tokens_and_bytes():
    tok = train_fast(TEXT, 300)
    h = length_hist(tok, TEXT)
    assert sum(h["hist"].values()) == h["tokens"] == len(tok.encode(TEXT))
    assert sum(k * v for k, v in h["hist"].items()) == len(TEXT.encode())


def test_unseen_text_is_mostly_bytes():
    tok = train_fast(TEXT, 300)
    assert length_hist(tok, "qxzjvk")["byte_fallback_frac"] == 1.0
    assert length_hist(tok, "")["tokens"] == 0
