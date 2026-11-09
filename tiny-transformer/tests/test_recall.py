import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from tinygpt import GPT, GPTConfig  # noqa: E402
from tinygpt.recall import make_batch, recall_accuracy, seq_len, vocab_size  # noqa: E402
import recall_bench as rb  # noqa: E402


def test_batch_shapes_and_vocab_ranges():
    x, y, a = make_batch(16, 3, 6, 5, np.random.default_rng(0))
    assert x.shape == (16, seq_len(3) - 1) and y.shape == x.shape
    assert x.max() < vocab_size(6, 5) and (a >= 6).all()


def test_keys_distinct_and_answers_correct():
    rng = np.random.default_rng(1)
    x, y, a = make_batch(64, 4, 8, 8, rng)
    full = np.concatenate([x, y[:, -1:]], axis=1)
    for row, ans in zip(full, a):
        keys, vals = row[0:8:2], row[1:8:2]
        assert len(set(keys)) == 4
        table = dict(zip(keys, vals))
        for k, v in zip(row[8::2], row[9::2]):
            assert table[k] == v
        assert row[-1] == ans


def test_requires_enough_keys():
    try:
        make_batch(1, 5, 4, 4, np.random.default_rng(0))
    except AssertionError:
        return
    raise AssertionError("expected assertion")


def test_untrained_model_near_chance_and_training_helps():
    cfg = GPTConfig(vocab_size(4, 4), seq_len(1) - 1, 1, 1, 16, rope=True)
    m = GPT(cfg, seed=0)
    assert recall_accuracy(m, 1, 4, 4, n=200) < 0.6


def test_bench_table_format():
    rows = {(k, n): [0.5, 0.7] for k in ("learned", "rope") for n in (2, 4)}
    t = rb.fmt(rows, [2, 4])
    assert "| rope |" in t and "0.60 (min 0.50)" in t
