import pytest
from bpekit import train, merge_tree, merge_depth, format_tree

TEXT = "the quick brown fox jumps over the lazy dog. " * 30


def leaves(n):
    return [n[0]] if n[1] is None else leaves(n[1]) + leaves(n[2])


def test_leaves_spell_token_bytes():
    t = train(TEXT, 300)
    for i in range(256, 256 + len(t.merges)):
        assert bytes(leaves(merge_tree(t, i))) == t.vocab[i]


def test_depth_and_byte_leaf():
    t = train(TEXT, 300)
    assert merge_depth(t, 65) == 0
    assert merge_tree(t, 65) == (65, None, None)
    assert merge_depth(t, 256) == 1
    assert all(merge_depth(t, i) >= 1 for i in range(256, 256 + len(t.merges)))


def test_invalid_and_format():
    t = train(TEXT, 300)
    with pytest.raises(ValueError):
        merge_tree(t, 256 + len(t.merges))
    out = format_tree(t, 256 + len(t.merges) - 1)
    assert out.splitlines()[0].endswith(f"(id {256 + len(t.merges) - 1})")
    assert len(out.splitlines()) >= 3
