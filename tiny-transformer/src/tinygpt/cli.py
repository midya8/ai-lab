"""Use a self-contained checkpoint:
  python -m tinygpt.cli generate ckpt.npz --prompt "To be" -n 200
  python -m tinygpt.cli eval ckpt.npz --data data/corpus.txt
"""
from __future__ import annotations

import argparse

import numpy as np

from .checkpoint import evaluate, evaluate_by_position, evaluate_sliding, load_checkpoint
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
    g.add_argument("--top-p", type=float, default=None, help="nucleus sampling mass in (0,1)")
    g.add_argument("--repetition-penalty", type=float, default=1.0, help=">1 discourages repeating seen tokens")
    g.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("ckpt")
    e.add_argument("--data", required=True)
    e.add_argument("--tail", type=float, default=1.0, help="evaluate only the last fraction of the file")
    e.add_argument("--by-position", type=int, default=0, metavar="BINS",
                   help="also print mean loss per context-position bin")
    e.add_argument("--stride", type=int, default=0,
                   help="also print sliding-window perplexity with this stride (smaller = more context)")
    a = ap.parse_args(argv)

    model, tok = load_checkpoint(a.ckpt)
    if a.cmd == "generate":
        ids = tok.encode(a.prompt) if a.prompt else [0]
        out = generate_cached(model, ids, a.n, temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, seed=a.seed,
                              repetition_penalty=a.repetition_penalty)
        print(tok.decode(out))
    else:
        text = open(a.data, encoding="utf-8").read()
        text = text[int(len(text) * (1 - a.tail)):]
        r = evaluate(model, np.array(tok.encode(text)))
        print(f"loss {r['loss']:.4f}  perplexity {r['perplexity']:.3f}  tokens {r['tokens']}")
        if a.by_position:
            bp = evaluate_by_position(model, np.array(tok.encode(text)), bins=a.by_position)
            for b in bp["bins"]:
                print(f"  pos {b['start']:>4}-{b['end']:<4} loss {b['loss']:.4f}")
            print(f"  context gain (first - last bin): {bp['gain']:.4f}")
        if a.stride:
            sw = evaluate_sliding(model, np.array(tok.encode(text)), stride=a.stride)
            print(f"sliding(stride={sw['stride']}) loss {sw['loss']:.4f}  perplexity {sw['perplexity']:.3f}  tokens {sw['tokens']}")


if __name__ == "__main__":
    main()
