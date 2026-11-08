"""RoPE vs learned absolute positions: held-out loss, and extrapolation past the training window.

    PYTHONPATH=src python experiments/rope_vs_learned.py [--steps 400] [--block 32]

Both models share every hyper-parameter and seed; they differ only in how positions enter.
Held-out loss is measured on the corpus tail with non-overlapping windows of several lengths.
A learned-position model has no embedding past `block_size`, so those cells are "n/a";
RoPE can be run on longer windows, and whether it *helps* there is an empirical question,
which this script measures rather than assumes (tiny models on a 2.6k-char corpus can
degrade on unseen lengths).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, Adam, CharTokenizer  # noqa: E402
from tinygpt.train import get_batch  # noqa: E402

HERE = os.path.dirname(__file__)


def window_loss(model, data, length, batch=16):
    n = (len(data) - 1) // length
    if n < 1:
        return None
    xs = np.stack([data[i * length:(i + 1) * length] for i in range(n)])
    ys = np.stack([data[i * length + 1:(i + 1) * length + 1] for i in range(n)])
    tot = 0.0
    for s in range(0, n, batch):
        tot += model.forward(xs[s:s + batch], ys[s:s + batch])[1] * xs[s:s + batch].size
    return tot / xs.size


def train(rope, data, vocab, args):
    m = GPT(GPTConfig(vocab, args.block, args.layers, args.heads, args.embd, rope=rope), seed=args.seed)
    opt = Adam(m.params, lr=args.lr)
    rng = np.random.default_rng(args.seed)
    for _ in range(args.steps):
        x, y = get_batch(data, args.batch, args.block, rng)
        _, g = m.loss_and_grads(x, y)
        opt.step(g)
    return m


def run(args):
    text = open(os.path.join(HERE, "..", "data", "corpus.txt"), encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    cut = int(len(data) * 0.85)
    train_d, val = data[:cut], data[cut:]
    rows = {}
    for name, rope in (("learned", False), ("rope", True)):
        m = train(rope, train_d, tok.vocab_size, args)
        rows[name] = {"params": m.num_params(), "loss": {}}
        for L in args.lengths:
            if not rope and L > args.block:
                rows[name]["loss"][L] = None
            else:
                rows[name]["loss"][L] = window_loss(m, val, L)
    return rows


def fmt(rows, lengths):
    out = ["| positions | params | " + " | ".join(f"len {L}" for L in lengths) + " |",
           "|---|---|" + "---|" * len(lengths)]
    for name, r in rows.items():
        cells = ["n/a" if r["loss"][L] is None else f"{r['loss'][L]:.3f}" for L in lengths]
        out.append(f"| {name} | {r['params']:,} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 96])
    args = ap.parse_args(argv)
    print(f"held-out loss (nats/char), trained with block_size={args.block}, {args.steps} steps")
    print(fmt(run(args), args.lengths))


if __name__ == "__main__":
    main()
