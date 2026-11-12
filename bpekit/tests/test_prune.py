from bpekit.bpe import train_fast
from bpekit.prune import prune

TEXT = "the cat sat on the mat. " * 50 + "zebra quartz " * 5


def test_prune_shrinks_and_roundtrips():
    tok = train_fast(TEXT, 300)
    small = prune(tok, "the cat sat on the mat. " * 10, min_count=1)
    assert len(small.merges) < len(tok.merges)
    for s in [TEXT, "zebra", "héllo wörld"]:
        assert small.decode(small.encode(s)) == s


def test_same_length_on_reference():
    tok = train_fast(TEXT, 300)
    ref = "the cat sat on the mat. " * 10
    small = prune(tok, ref)
    assert len(small.encode(ref)) == len(tok.encode(ref))


def test_dependencies_kept_and_valid():
    tok = train_fast(TEXT, 300)
    small = prune(tok, TEXT, min_count=5)
    for n, (a, b) in enumerate(small.merges):
        assert a < 256 + n and b < 256 + n


def test_specials_preserved():
    tok = train_fast(TEXT, 290)
    small = prune(tok, "cat")
    assert list(small.specials) == list(tok.specials)
