from bpekit.bpe import train_fast
from bpekit.stream import StreamDecoder, stream_decode

TEXT = "héllo wörld ☃ snow 日本語 テスト " * 30


def test_concat_equals_decode():
    tok = train_fast(TEXT, 300)
    ids = tok.encode(TEXT)
    assert "".join(stream_decode(tok, ids)) == tok.decode(ids)


def test_never_emits_replacement_midway():
    tok = train_fast(TEXT, 260)  # few merges: characters split across tokens
    ids = tok.encode("☃日本語")
    assert any(len(tok.vocab[i]) < 3 for i in ids)
    pieces = list(stream_decode(tok, ids))
    assert "\ufffd" not in "".join(pieces)
    assert "".join(pieces) == "☃日本語"


def test_mid_character_returns_empty():
    tok = train_fast(TEXT, 256)
    d = StreamDecoder(tok)
    b = "☃".encode()
    assert d.push(b[0]) == "" and d.push(b[1]) == ""
    assert d.push(b[2]) == "☃"


def test_truncated_stream_flushes_replacement():
    tok = train_fast(TEXT, 256)
    d = StreamDecoder(tok)
    d.push("☃".encode()[0])
    assert d.flush() == "\ufffd"


def test_specials():
    tok = train_fast(TEXT, 270)
    sid = next(iter(tok.specials.values()))
    assert "".join(stream_decode(tok, [65, sid])) == tok.decode([65, sid])
