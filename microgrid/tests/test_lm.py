import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data, train, generate, evaluate

TEXT = "abcabcabd " * 60


def test_training_reduces_val_loss():
    chars, tr, va = make_data(TEXT)
    m = MiniGPT(len(chars), d=16, heads=2, layers=1, ctx=8, seed=0)
    before = evaluate(m, va, 8)
    hist = train(m, tr, va, 8, steps=60, bs=8, lr=1e-2, log_every=30)
    assert hist[-1][2] < before * 0.6


def test_generate_length_and_alphabet():
    chars, tr, va = make_data(TEXT)
    m = MiniGPT(len(chars), d=16, heads=2, layers=1, ctx=8)
    out = generate(m, chars, tr[:4], 10, 8)
    assert len(out) == 14 and set(out) <= set(chars)
