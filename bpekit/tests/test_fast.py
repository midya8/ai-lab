import random
from bpekit import train, train_fast


def _corpus(seed, n=4000):
    r = random.Random(seed)
    vocab = ["".join(r.choice("abcdeé日 ") for _ in range(r.randint(1, 7))) for _ in range(60)]
    return " ".join(r.choice(vocab) for _ in range(n)) + "\n\n x1 22 ,.;"


def test_identical_merges_to_reference():
    for seed in range(4):
        t = _corpus(seed)
        for v in (260, 330, 600):
            assert train(t, v).merges == train_fast(t, v).merges


def test_stops_when_no_repeated_pair_and_roundtrips():
    t = train_fast("ab cd ef", 400)
    assert t.merges == [] and t.decode(t.encode("ab cd ef")) == "ab cd ef"


def test_rejects_small_vocab():
    import pytest
    with pytest.raises(ValueError):
        train_fast("abc", 100)
