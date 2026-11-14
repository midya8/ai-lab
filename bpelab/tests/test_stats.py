import math
import pytest
from bpelab import BPETokenizer
from bpelab.stats import token_stats


class Fake:
    vocab_size = 4

    def encode(self, text):
        return [0, 1, 2, 3] * 2


def test_uniform_distribution_has_efficiency_one():
    s = token_stats(Fake(), "hello world")
    assert s["tokens"] == 8
    assert s["fertility"] == 4.0
    assert s["vocab_utilisation"] == 1.0
    assert math.isclose(s["renyi_efficiency"], 1.0)


def test_skewed_is_lower_and_bytes_per_token():
    class Skew(Fake):
        def encode(self, text):
            return [0] * 7 + [1]
    s = token_stats(Skew(), "abcd efgh")
    assert s["renyi_efficiency"] < 0.5
    assert math.isclose(s["bytes_per_token"], 9 / 8)


def test_real_bpe_fertility_drops_with_vocab():
    text = "the quick brown fox jumps over the lazy dog. " * 40
    small = token_stats(BPETokenizer.train(text, 270), text)
    big = token_stats(BPETokenizer.train(text, 330), text)
    assert big["fertility"] < small["fertility"]


def test_empty_raises():
    with pytest.raises(ValueError):
        token_stats(Fake(), "   ")


def test_dead_vocab_counts_and_top():
    from bpelab.stats import dead_vocab

    class T:
        vocab_size = 5

        def encode(self, text):
            return [0, 0, 0, 2]
    d = dead_vocab(T(), "x")
    assert d["dead_ids"] == [1, 3, 4]
    assert math.isclose(d["dead_fraction"], 0.6)
    assert d["top"][0] == (0, 3)


def test_dead_vocab_heldout_bpe_has_dead_tokens():
    from bpelab.stats import dead_vocab
    train = "the quick brown fox jumps over the lazy dog. " * 30 + "zebra quartz " * 20
    tok = BPETokenizer.train(train, 330)
    d = dead_vocab(tok, "the dog. the fox. " * 5)
    assert d["dead_fraction"] > 0.5
    assert dead_vocab(tok, train)["dead_fraction"] < d["dead_fraction"]


def test_dead_vocab_empty_raises():
    from bpelab.stats import dead_vocab
    with pytest.raises(ValueError):
        dead_vocab(Fake.__mro__[0]() if False else type("E", (), {"vocab_size": 3, "encode": lambda s, t: []})(), "")


def test_compare_tokenizers_sorted_and_named():
    from bpelab.stats import compare_tokenizers
    text = "the cat sat on the mat. the cat ran. " * 20
    small = BPETokenizer.train(text, 260)
    big = BPETokenizer.train(text, 300)
    rows = compare_tokenizers({"small": small, "big": big}, text)
    assert [r["name"] for r in rows][0] == "big"
    assert rows[0]["fertility"] <= rows[1]["fertility"]


def test_compare_tokenizers_empty_raises():
    from bpelab.stats import compare_tokenizers
    with pytest.raises(ValueError):
        compare_tokenizers({}, "x y")


def test_transfer_matrix_in_domain_beats_out_of_domain():
    from bpelab import transfer_matrix
    a = "the cat sat on the mat and the cat ran " * 30
    b = "xyzzy quux plugh frobnicate zork wibble " * 30
    ta, tb = BPETokenizer.train(a, 300), BPETokenizer.train(b, 300)
    m = transfer_matrix({"A": ta, "B": tb}, {"a": a, "b": b})
    assert m["A"]["a"] > m["A"]["b"]
    assert m["B"]["b"] > m["B"]["a"]


def test_transfer_matrix_validates():
    from bpelab import transfer_matrix
    with pytest.raises(ValueError):
        transfer_matrix({}, {"a": "x"})
