from bpekit.bpe import train_fast
from bpekit.variants import boundary_variants

TEXT = "the cat sat on the mat. " * 50


def test_rows_roundtrip_and_sorted():
    tok = train_fast(TEXT, 300)
    rows = boundary_variants(tok, TEXT)
    assert {r["word"] for r in rows} == {"the", "cat", "sat", "on", "mat"}
    for r in rows:
        assert tok.decode(r["bare"]) == r["word"]
        assert tok.decode(r["spaced"]) == " " + r["word"]
        assert r["extra_tokens"] == len(r["spaced"]) - len(r["bare"])
    extras = [r["extra_tokens"] for r in rows]
    assert extras == sorted(extras, reverse=True)


def test_min_count_and_limit():
    tok = train_fast(TEXT, 300)
    assert {r["word"] for r in boundary_variants(tok, TEXT, min_count=100)} == {"the"}
    assert len(boundary_variants(tok, TEXT, limit=2)) == 2
    assert boundary_variants(tok, "") == []
