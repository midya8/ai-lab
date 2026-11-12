import pytest
from bpekit import train, encode_with_offsets, chunk_by_tokens

TEXT = ("the quick brown fox jumps over the lazy dog. " * 30) + "naïve café 日本語 123 456\n\n  end"


def tok():
    return train(TEXT, 300)


def test_offsets_match_encode_and_cover_text():
    t = tok()
    s = "the quick fox, café 日本語!"
    ids, spans = encode_with_offsets(t, s)
    assert ids == t.encode(s)
    assert spans[0][0] == 0 and spans[-1][1] == len(s)
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert 0 <= a < b <= len(s)


def test_ascii_spans_reconstruct_exactly():
    t = tok()
    s = "the lazy dog jumps over the fox"
    ids, spans = encode_with_offsets(t, s)
    assert "".join(s[a:b] for a, b in spans) == s
    for i, (a, b) in zip(ids, spans):
        assert t.vocab[i].decode() == s[a:b]


def test_special_token_span():
    t = tok()
    SP = next(iter(t.specials))
    s = "hi" + SP + "yo"
    ids, spans = encode_with_offsets(t, s)
    k = ids.index(t.specials[SP])
    assert s[spans[k][0]:spans[k][1]] == SP


def test_chunks_respect_budget_and_concat():
    t = tok()
    chunks = chunk_by_tokens(t, TEXT, 40)
    assert "".join(chunks) == TEXT
    assert all(len(t.encode(c)) <= 40 for c in chunks)
    assert len(chunks) > 3


def test_chunk_overlap_and_validation():
    t = tok()
    chunks = chunk_by_tokens(t, TEXT, 40, overlap=10)
    assert len(chunks) > len(chunk_by_tokens(t, TEXT, 40))
    assert all(len(t.encode(c)) <= 40 for c in chunks)
    assert chunks[1].split()[0] in chunks[0]
    with pytest.raises(ValueError):
        chunk_by_tokens(t, TEXT, 5, overlap=5)


def test_oversized_word_gets_own_chunk():
    t = tok()
    long = "x" * 50
    chunks = chunk_by_tokens(t, "a " + long + " b", 3)
    assert long in "".join(chunks) and any(c.strip() == long for c in chunks)
