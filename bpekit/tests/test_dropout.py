import random
import pytest
from bpekit import train_fast

TEXT = "the quick brown fox jumps over the lazy dog. " * 40 + "héllo 日本語 world"


def tok():
    return train_fast(TEXT, 330)


def test_zero_dropout_is_deterministic_default():
    t = tok()
    assert t.encode(TEXT, dropout=0.0) == t.encode(TEXT)


def test_dropout_roundtrips_and_varies():
    t = tok(); rng = random.Random(0)
    outs = {tuple(t.encode(TEXT, dropout=0.3, rng=rng)) for _ in range(5)}
    assert len(outs) > 1
    for o in outs:
        assert t.decode(list(o)) == TEXT


def test_dropout_lengthens_sequences_monotonically_on_average():
    t = tok()
    lens = []
    for p in (0.0, 0.2, 0.5, 0.9):
        rng = random.Random(1)
        lens.append(sum(len(t.encode(TEXT, dropout=p, rng=rng)) for _ in range(5)) / 5)
    assert lens == sorted(lens) and lens[-1] > lens[0]


def test_seeded_rng_reproducible_and_validation():
    t = tok()
    a = t.encode(TEXT, 0.4, random.Random(7)); b = t.encode(TEXT, 0.4, random.Random(7))
    assert a == b
    with pytest.raises(ValueError):
        t.encode("x", dropout=1.0)


def test_specials_preserved_under_dropout():
    t = tok(); sp = next(iter(t.specials)); s = "a" + sp + "b"
    ids = t.encode(s, dropout=0.5, rng=random.Random(2))
    assert t.specials[sp] in ids and t.decode(ids) == s
