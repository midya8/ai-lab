"""Train a char-level GPT:  python -m tinygpt.train --data data/shakespeare_snippet.txt"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .model import GPT, GPTConfig
from .optim import Adam
from .tokenizer import CharTokenizer


def get_batch(data, B, T, rng):
    ix = rng.integers(0, len(data) - T - 1, B)
    x = np.stack([data[i:i + T] for i in ix])
    y = np.stack([data[i + 1:i + T + 1] for i in ix])
    return x, y


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/corpus.txt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=48)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="tinygpt.npz")
    ap.add_argument("--sample", type=int, default=200)
    a = ap.parse_args(argv)

    text = open(a.data, encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    model = GPT(GPTConfig(tok.vocab_size, a.block, a.layers, a.heads, a.embd))
    opt = Adam(model.params, lr=a.lr)
    rng = np.random.default_rng(0)
    print(f"vocab={tok.vocab_size} params={model.num_params():,} chars={len(text):,}")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y = get_batch(data, a.batch, a.block, rng)
        loss, grads = model.loss_and_grads(x, y)
        opt.step(grads)
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} loss {loss:.3f} ({time.time() - t0:.0f}s)")
    model.save(a.out)
    seed = tok.encode(text[:8])
    print("--- sample ---")
    print(tok.decode(model.generate(seed, a.sample, temperature=0.8, top_k=10)))


if __name__ == "__main__":
    main()
