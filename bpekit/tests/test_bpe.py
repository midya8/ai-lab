import pytest
from bpekit import train, BPETokenizer

TEXT = ("the quick brown fox jumps over the lazy dog. " * 30) + "naïve café 日本語 123 456\n\n  end"


def test_roundtrip_unicode_and_whitespace():
    t = train(TEXT, 300)
    for s in [TEXT, "", "  ", "héllo wörld 😀", "a\n\n\nb\t c"]:
        assert t.decode(t.encode(s)) == s


def test_compresses_training_text():
    t = train(TEXT, 300)
    assert t.compression(TEXT) > 2.0
    assert len(t.encode(TEXT)) < len(TEXT.encode())


def test_vocab_size_bound_and_deterministic():
    a, b = train(TEXT, 280), train(TEXT, 280)
    assert a.merges == b.merges and len(a) <= 281


def test_specials_atomic():
    t = train(TEXT, 290)
    ids = t.encode("hi<|endoftext|>there")
    assert t.specials["<|endoftext|>"] in ids
    assert t.decode(ids) == "hi<|endoftext|>there"


def test_save_load(tmp_path):
    t = train(TEXT, 290); p = tmp_path / "t.json"; t.save(p)
    u = BPETokenizer.load(p)
    assert u.encode(TEXT) == t.encode(TEXT)


def test_bad_vocab():
    with pytest.raises(ValueError):
        train("x", 100)


def test_larger_vocab_never_longer():
    assert len(train(TEXT, 320).encode(TEXT)) <= len(train(TEXT, 270).encode(TEXT))
