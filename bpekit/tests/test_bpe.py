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


def test_cli_train_encode_decode_eval(tmp_path, capsys):
    from bpekit.__main__ import main
    f = tmp_path / "c.txt"; f.write_text(TEXT * 3, encoding="utf-8")
    tok = str(tmp_path / "t.json")
    assert main(["train", str(f), "-v", "290", "-o", tok]) == 0
    main(["encode", tok, "the lazy dog"])
    ids = capsys.readouterr().out.split()[-1:] and capsys.readouterr().out
    main(["encode", tok, "the lazy dog"])
    ids = [int(x) for x in capsys.readouterr().out.split()]
    main(["decode", tok, *map(str, ids)])
    assert capsys.readouterr().out.strip() == "the lazy dog"
    main(["eval", str(f), "-v", "290"])
    assert "held-out" in capsys.readouterr().out
