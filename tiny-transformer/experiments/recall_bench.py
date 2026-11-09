"""Key-value recall benchmark: learned positions vs RoPE, and accuracy vs number of stored pairs.

    PYTHONPATH=src python experiments/recall_bench.py [--seeds 0 1] [--steps 1500] [--pairs 2 4 6]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, Adam  # noqa: E402
from tinygpt.recall import make_batch, recall_accuracy, seq_len, vocab_size  # noqa: E402


def train(n_pairs, args, rope, seed):
    cfg = GPTConfig(vocab_size(args.keys, args.vals), seq_len(n_pairs), args.layers, args.heads, args.embd, rope=rope)
    m = GPT(cfg, seed=seed)
    opt = Adam(m.params, lr=args.lr)
    rng = np.random.default_rng(seed)
    for _ in range(args.steps):
        x, y, _ = make_batch(args.batch, n_pairs, args.keys, args.vals, rng)
        _, g = m.loss_and_grads(x, y)
        opt.step(g)
    return recall_accuracy(m, n_pairs, args.keys, args.vals)


def run(args):
    rows = {}
    for rope in (False, True):
        for n in args.pairs:
            rows[("rope" if rope else "learned", n)] = [train(n, args, rope, s) for s in args.seeds]
    return rows


def fmt(rows, pairs):
    out = ["| positions | " + " | ".join(f"{n} pairs" for n in pairs) + " |", "|---|" + "---|" * len(pairs)]
    for kind in ("learned", "rope"):
        cells = [f"{np.mean(rows[(kind, n)]):.2f} (min {min(rows[(kind, n)]):.2f})" for n in pairs]
        out.append(f"| {kind} | " + " | ".join(cells) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--pairs", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--vals", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    a = ap.parse_args()
    r = run(a)
    t = fmt(r, a.pairs)
    print(t)
    with open(os.path.join(os.path.dirname(__file__), "recall_results.md"), "w") as f:
        f.write(f"steps={a.steps} seeds={a.seeds} chance={1/a.vals:.3f}\n\n{t}\n")
