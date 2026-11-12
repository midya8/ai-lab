from bpekit import train_fast, fertility

TXT = "the cat sat on the mat. the dog ate the bone. " * 30


def test_empty_text():
    tok = train_fast(TXT, 300)
    assert fertility(tok, "...")["words"] == 0


def test_in_domain_beats_foreign_text():
    tok = train_fast(TXT, 300)
    a = fertility(tok, TXT)
    b = fertility(tok, "zyxwv qwrtp xkcdz " * 10)
    assert a["mean"] < b["mean"]
    assert a["single_frac"] > b["single_frac"]


def test_worst_words_and_counts():
    tok = train_fast(TXT, 300)
    r = fertility(tok, "the the the qzxjkv")
    assert r["words"] == 4
    assert r["worst"][0][0] == "qzxjkv"
    assert r["worst"][0][1] == len(tok.encode("qzxjkv"))
    assert r["p95"] >= 1 and r["mean"] >= 1
