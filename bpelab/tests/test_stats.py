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
