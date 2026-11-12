import pytest
from bpekit.bpe import train_fast
from bpekit import pack

DOCS = ["héllo wörld ☃ snow", "the quick brown fox", "日本語 テスト"] * 6
TEXT = " ".join(DOCS) * 5


def test_rows_exact_length_and_lossless_stream():
    tok = train_fast(TEXT, 290)
    rows, _ = pack(tok, DOCS, 16, drop_last=False)
    assert all(len(r) == 16 for r in rows[:-1])
    flat = [i for r in rows for i in r]
    eos = next(iter(tok.specials.values()))
    expect = []
    for d in DOCS:
        expect += tok.encode(d) + [eos]
    assert flat == expect


def test_drop_last_and_doc_starts():
    tok = train_fast(TEXT, 290)
    rows, starts = pack(tok, DOCS, 16)
    assert all(len(r) == 16 for r in rows)
    eos = next(iter(tok.specials.values()))
    for r, ss in zip(rows, starts):
        for s in ss:
            assert s == 0 or r[s - 1] == eos
            assert 0 <= s < 16
    assert starts[0][0] == 0


def test_bad_seq_len():
    tok = train_fast(TEXT, 290)
    with pytest.raises(ValueError):
        pack(tok, DOCS, 0)
