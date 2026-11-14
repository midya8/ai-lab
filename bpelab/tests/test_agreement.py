import pytest
from bpelab import BPETokenizer, UnigramTokenizer, segmentation_agreement

TEXT = "the cat sat on the mat the cat ate the rat while running and jumping " * 30


def test_self_agreement_is_perfect():
    tok = BPETokenizer.train(TEXT, 300)
    r = segmentation_agreement(tok, tok, TEXT)
    assert r["identical"] == 1.0 and r["f1"] == 1.0


def test_different_tokenizers_disagree_somewhere():
    a = BPETokenizer.train(TEXT, 270)
    b = BPETokenizer.train(TEXT, 400)
    r = segmentation_agreement(a, b, TEXT)
    assert 0.0 <= r["identical"] < 1.0
    assert 0.0 <= r["f1"] <= 1.0
    # symmetry: swapping swaps precision and recall
    r2 = segmentation_agreement(b, a, TEXT)
    assert r["precision"] == pytest.approx(r2["recall"])
    assert r["identical"] == pytest.approx(r2["identical"])


def test_empty_text_raises():
    tok = BPETokenizer.train(TEXT, 300)
    with pytest.raises(ValueError):
        segmentation_agreement(tok, tok, "!!!")
