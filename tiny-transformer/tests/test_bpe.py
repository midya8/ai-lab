import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import BPETokenizer  # noqa: E402

TEXT = "the cat sat on the mat. the cat ate the rat. " * 20


def test_roundtrip_ascii_and_unicode():
    tok = BPETokenizer.train(TEXT, 300)
    for s in [TEXT, "unseen words here", "héllo wörld ✓ 日本語", ""]:
        assert tok.decode(tok.encode(s)) == s


def test_compresses():
    tok = BPETokenizer.train(TEXT, 300)
    assert len(tok.encode(TEXT)) < len(TEXT.encode()) / 3
    assert tok.vocab_size > 256


def test_deterministic_and_save_load(tmp_path):
    a = BPETokenizer.train(TEXT, 290)
    b = BPETokenizer.train(TEXT, 290)
    assert a.merges == b.merges
    p = str(tmp_path / "t.json")
    a.save(p)
    c = BPETokenizer.load(p)
    assert c.encode(TEXT) == a.encode(TEXT)


def test_stops_when_no_repeats():
    tok = BPETokenizer.train("abcdefg", 400)
    assert tok.vocab_size == 256


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train("aaabdaaabac", 257)
    assert tok.merges == [(97, 97)]
