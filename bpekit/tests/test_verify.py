from bpekit.bpe import train_fast
from bpekit import verify, fuzz_strings

TEXT = "héllo wörld ☃ snow the quick brown fox 日本語 " * 20


def test_trained_tokenizer_round_trips_fuzz():
    tok = train_fast(TEXT, 300)
    rep = verify(tok, ["plain text", "日本語 ☃"], n_fuzz=150)
    assert rep["checked"] == 2 * (2 + 151)
    assert rep["failures"] == []


def test_detects_a_broken_decoder():
    tok = train_fast(TEXT, 300)
    orig = tok.decode
    tok.decode = lambda ids: orig(ids).replace("o", "0")
    rep = verify(tok, ["foo"], n_fuzz=0)
    assert rep["failures"] and rep["failures"][0]["got"] == "f00"


def test_fuzz_is_deterministic():
    assert fuzz_strings(10, seed=3) == fuzz_strings(10, seed=3)
