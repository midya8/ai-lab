from bpekit.bpe import train_fast
from bpekit.coverage import coverage

TEXT = "the cat sat on the mat. " * 50 + "zebra quartz " * 5


def test_full_k_covers_everything():
    tok = train_fast(TEXT, 290)
    c = coverage(tok, TEXT, ks=(1, 10**6))
    assert c["at"][10**6] == 1.0
    assert c["tokens"] == len(tok.encode(TEXT))


def test_monotone_and_thresholds():
    tok = train_fast(TEXT, 290)
    c = coverage(tok, TEXT, ks=(1, 5, 20))
    assert c["at"][1] <= c["at"][5] <= c["at"][20]
    assert 1 <= c["k90"] <= c["k99"] <= c["distinct"]


def test_empty_text():
    tok = train_fast(TEXT, 290)
    c = coverage(tok, "")
    assert c["tokens"] == 0 and c["k90"] == 0 and c["at"][10] == 0.0
