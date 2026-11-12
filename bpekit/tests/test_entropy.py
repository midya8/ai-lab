import math
from bpekit.bpe import train_fast
from bpekit.entropy import token_entropy

TEXT = "the cat sat on the mat. " * 50


def test_empty_and_single_token():
    tok = train_fast(TEXT, 300)
    assert token_entropy(tok, "")["tokens"] == 0
    r = token_entropy(tok, "a")
    assert r["entropy_bits"] == 0.0 and r["efficiency"] == 1.0


def test_uniform_bytes_entropy():
    tok = train_fast(TEXT, 256 + 1)  # essentially byte-level
    r = token_entropy(tok, "abcd" * 10)
    assert r["distinct"] >= 2
    assert 0 < r["efficiency"] <= 1.0 + 1e-9
    assert r["entropy_bits"] <= math.log2(r["distinct"]) + 1e-9


def test_bigger_vocab_lowers_bits_per_byte_vs_raw():
    tok = train_fast(TEXT, 300)
    r = token_entropy(tok, TEXT)
    assert r["bits_per_byte"] < 8.0
