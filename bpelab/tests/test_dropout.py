import random
from bpelab import BPETokenizer

TEXT = "the quick brown fox jumps over the lazy dog. " * 40 + "unbelievably unbelievable believers"


def tok():
    return BPETokenizer.train(TEXT, 300, special_tokens=["<eos>"])


def test_dropout_zero_is_deterministic():
    t = tok()
    assert t.encode(TEXT) == t.encode(TEXT, dropout=0.0)


def test_dropout_roundtrips_losslessly():
    t = tok()
    rng = random.Random(1)
    s = "unbelievable fox <eos> naïve café 123"
    for _ in range(20):
        assert t.decode(t.encode(s, dropout=0.5, rng=rng)) == s


def test_dropout_varies_and_lengthens():
    t = tok()
    rng = random.Random(2)
    base = len(t.encode(TEXT))
    segs = {tuple(t.encode("unbelievably", dropout=0.4, rng=rng)) for _ in range(30)}
    assert len(segs) > 1
    avg = sum(len(t.encode(TEXT, dropout=0.3, rng=rng)) for _ in range(5)) / 5
    assert avg > base


def test_dropout_one_gives_bytes():
    t = tok()
    s = "hello world"
    assert t.encode(s, dropout=1.0, rng=random.Random(0)) == list(s.encode())
