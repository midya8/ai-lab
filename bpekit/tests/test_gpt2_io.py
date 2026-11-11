import json
import pytest
from bpekit import train, export_gpt2, import_gpt2, bytes_to_unicode

TEXT = ("the quick brown fox jumps over the lazy dog. " * 30) + "naïve café 日本語 123 456\n\n  end"


def test_byte_map_is_bijection_and_printable():
    m = bytes_to_unicode()
    assert len(m) == 256 and len(set(m.values())) == 256
    assert all(c.isprintable() and not c.isspace() for c in m.values())


def test_roundtrip_identical_encoding(tmp_path):
    t = train(TEXT, 320)
    v, m = tmp_path / "vocab.json", tmp_path / "merges.txt"
    export_gpt2(t, v, m)
    t2 = import_gpt2(v, m, specials=tuple(t.specials))
    assert t2.merges == t.merges and t2.specials == t.specials
    for s in [TEXT, "héllo 😀", "<|endoftext|>x"]:
        assert t2.encode(s) == t.encode(s)


def test_vocab_file_format(tmp_path):
    t = train(TEXT, 300)
    v, m = tmp_path / "v.json", tmp_path / "m.txt"
    export_gpt2(t, v, m)
    vocab = json.loads(v.read_text(encoding="utf-8"))
    assert len(vocab) == len(t) and vocab["Ġthe"] == t.encode(" the")[0] or "Ġthe" not in vocab
    assert m.read_text(encoding="utf-8").startswith("#version")
    assert len(m.read_text(encoding="utf-8").splitlines()) == 1 + len(t.merges)


def test_import_rejects_inconsistent_vocab(tmp_path):
    t = train(TEXT, 300)
    v, m = tmp_path / "v.json", tmp_path / "m.txt"
    export_gpt2(t, v, m)
    d = json.loads(v.read_text(encoding="utf-8"))
    k = [k for k, i in d.items() if i == 256][0]
    k2 = [k for k, i in d.items() if i == 257][0]
    d[k], d[k2] = 257, 256
    v.write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(ValueError):
        import_gpt2(v, m)
