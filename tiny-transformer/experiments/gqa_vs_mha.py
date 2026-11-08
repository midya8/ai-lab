"""Multi-head vs grouped-query vs multi-query attention: quality, parameters, KV-cache size.

    PYTHONPATH=src python experiments/gqa_vs_mha.py [--steps 400] [--heads 8] [--embd 64]

Every model shares seed, data and hyper-parameters; only the number of key/value heads differs.
The cache column is the inference KV cache for a full block_size context (float32).
Results at this scale are one seed on a ~2.6k-char corpus: differences of a few hundredths of a
nat are noise, so read the table as "does quality survive the cache savings", not as a ranking.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, Adam, CharTokenizer  # noqa: E402
from tinygpt.model import kv_cache_bytes  # noqa: E402
from tinygpt.checkpoint import evaluate  # noqa: E402
from tinygpt.train import get_batch  # noqa: E402

HERE = os.path.dirname(__file__)


def train(n_kv, data, vocab, args):
    cfg = GPTConfig(vocab, args.block, args.layers, args.heads, args.embd, n_kv_head=n_kv)
    m = GPT(cfg, seed=args.seed)
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
    rows = []
    for kv in args.kv_heads:
        m = train(kv, train_d, tok.vocab_size, args)
        rows.append({"kv_heads": m.cfg.kv_heads, "params": m.num_params(),
                     "cache_bytes": kv_cache_bytes(m.cfg, args.block),
                     "val_loss": evaluate(m, val)["loss"]})
    return rows


def fmt(rows, heads):
    base = rows[0]["cache_bytes"]
    out = ["| kv heads | variant | params | KV cache | cache vs MHA | val loss |", "|---|---|---|---|---|---|"]
    for r in rows:
        kind = "MHA" if r["kv_heads"] == heads else "MQA" if r["kv_heads"] == 1 else "GQA"
        out.append(f"| {r['kv_heads']} | {kind} | {r['params']:,} | {r['cache_bytes'] / 1024:.1f} KiB | "
                   f"{r['cache_bytes'] / base:.2f}x | {r['val_loss']:.3f} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kv-heads", type=int, nargs="+", default=[8, 4, 2, 1])
    args = ap.parse_args(argv)
    assert args.kv_heads[0] == args.heads, "first --kv-heads entry must equal --heads (the MHA baseline)"
    print(f"held-out loss (nats/char), {args.heads} query heads, {args.steps} steps, seed {args.seed}")
    print(fmt(run(args), args.heads))


if __name__ == "__main__":
    main()
