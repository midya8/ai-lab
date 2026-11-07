"""Use a self-contained checkpoint:
  python -m tinygpt.cli generate ckpt.npz --prompt "To be" -n 200
  python -m tinygpt.cli eval ckpt.npz --data data/corpus.txt
"""
from __future__ import annotations

import argparse

import numpy as np

from .checkpoint import evaluate, load_checkpoint
from .kvcache import generate_cached


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tinygpt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("ckpt")
    g.add_argument("--prompt", default="")
    g.add_argument("-n", type=int, default=200)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--data", required=True)
    e.add_argument("--tail", type=float, default=1.0, help="evaluate only the last fraction of the file")
    a = ap.parse_args(argv)

    model, tok = load_checkpoint(a.ckpt)
    if a.cmd == "generate":
        ids = tok.encode(a.prompt) if a.prompt else [0]
        out = generate_cached(model, ids, a.n, temperature=a.temperature, top_k=a.top_k, seed=a.seed)
        print(tok.decode(out))
    else:
        text = open(a.data, encoding="utf-8").read()
        text = text[int(len(text) * (1 - a.tail)):]
        r = evaluate(model, np.array(tok.encode(text)))
        print(f"loss {r['loss']:.4f}  perplexity {r['perplexity']:.3f}  tokens {r['tokens']}")


if __name__ == "__main__":
    main()
