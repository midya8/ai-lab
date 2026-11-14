import pytest
from bpelab import BPETokenizer, UnigramTokenizer, agreement_matrix

TEXT = "the cat sat on the mat the cat ate the rat while running and jumping " * 30


def test_matrix_symmetric_unit_diagonal():
    toks = {"bpe270": BPETokenizer.train(TEXT, 270), "bpe400": BPETokenizer.train(TEXT, 400),
            "uni": UnigramTokenizer.train(TEXT, 300)}
    m = agreement_matrix(toks, TEXT)
    for a in toks:
        assert m[a][a] == 1.0
        for b in toks:
            assert m[a][b] == pytest.approx(m[b][a])
            assert 0.0 <= m[a][b] <= 1.0


def test_single_tokenizer():
    t = BPETokenizer.train(TEXT, 300)
    assert agreement_matrix({"x": t}, TEXT) == {"x": {"x": 1.0}}
