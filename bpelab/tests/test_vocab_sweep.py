import pytest
from bpelab import vocab_size_sweep

TRAIN = "the quick brown fox jumps over the lazy dog. " * 40 + "pack my box with five dozen liquor jugs. " * 20
HELD = "the lazy dog jumps over the quick fox. pack the box. " * 5


def test_bytes_per_token_improves_with_vocab():
    r = vocab_size_sweep(TRAIN, HELD, [256, 280, 320])
    b = [x["bytes_per_token"] for x in r["rows"]]
    assert b[0] == pytest.approx(1.0) and b[1] > b[0] and b[2] >= b[1]
    assert r["rows"][0]["marginal_gain"] is None


def test_knee_found_when_gain_saturates():
    r = vocab_size_sweep(TRAIN, HELD, [256, 300, 400, 500], min_gain=0.05)
    assert r["knee"] in (300, 400)


def test_invalid_sizes():
    with pytest.raises(ValueError):
        vocab_size_sweep(TRAIN, HELD, [100])
