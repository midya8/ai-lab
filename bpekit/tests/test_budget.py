import pytest
from bpekit.bpe import train_fast
from bpekit import truncate

TEXT = "héllo wörld ☃ snow 日本語 テスト the quick brown fox " * 20


def test_fits_unchanged():
    tok = train_fast(TEXT, 300)
    assert truncate(tok, "snow", 50) == ("snow", len(tok.encode("snow")))


@pytest.mark.parametrize("keep", ["head", "tail", "middle"])
def test_budget_respected_and_clean(keep):
    tok = train_fast(TEXT, 262)
    for n in (1, 5, 17, 40):
        out, k = truncate(tok, TEXT, n, keep=keep)
        assert k <= n and len(tok.encode(out)) == k
        assert "\ufffd" not in out


def test_head_is_prefix_tail_is_suffix_and_middle_has_marker():
    tok = train_fast(TEXT, 300)
    h, _ = truncate(tok, TEXT, 20, "head")
    t, _ = truncate(tok, TEXT, 20, "tail")
    m, _ = truncate(tok, TEXT, 30, "middle", marker=" | ")
    assert TEXT.startswith(h) and TEXT.endswith(t) and " | " in m
    assert m.startswith(TEXT[:3]) and m.endswith(TEXT[-3:])


def test_bad_args():
    tok = train_fast(TEXT, 300)
    with pytest.raises(ValueError):
        truncate(tok, TEXT, 5, keep="x")
