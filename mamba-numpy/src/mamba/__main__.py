"""python -m mamba train|eval|demo ..."""
from __future__ import annotations

import argparse
import functools
import sys

import numpy as np

from .model import Mamba
from .tasks import associative_recall, score, selective_copy
from .train import fit


def _task(args):
    if args.task == "selective_copy":
        mk = functools.partial(selective_copy, seq_len=args.seq_len, n_data=args.n_data, vocab=args.vocab)
        return (lambda n, rng: mk(n, rng=rng)), args.vocab + 2
    n_keys = 8
    mk = functools.partial(associative_recall, n_pairs=args.n_data, n_keys=n_keys, vocab=args.vocab)
    return (lambda n, rng: mk(n, rng=rng)), n_keys + args.vocab + 1


def _add_task_args(p):
    p.add_argument("--task", choices=["selective_copy", "recall"], default="selective_copy")
    p.add_argument("--seq-len", type=int, default=16, help="noise length (selective_copy)")
    p.add_argument("--n-data", type=int, default=4, help="symbols to copy / key-value pairs")
    p.add_argument("--vocab", type=int, default=4)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mamba")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train a small Mamba-style model on a synthetic task")
    _add_task_args(t)
    t.add_argument("--lti", action="store_true", help="fixed (S4D-style) dynamics instead of selective")
    t.add_argument("--d-model", type=int, default=16)
    t.add_argument("--d-state", type=int, default=8)
    t.add_argument("--layers", type=int, default=2)
    t.add_argument("--steps", type=int, default=1500)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--save", help="checkpoint path (.npz)")
    e = sub.add_parser("eval", help="score a checkpoint (optionally at a different length)")
    _add_task_args(e)
    e.add_argument("checkpoint")
    e.add_argument("--n", type=int, default=1000)
    a = ap.parse_args(argv)

    make, vocab = _task(a)
    if a.cmd == "train":
        m = Mamba(vocab, a.d_model, a.d_state, a.layers, selective=not a.lti, seed=a.seed)
        print(f"{'LTI' if a.lti else 'selective'} model, {m.n_params()} params, state {m.state_bytes()} bytes/sequence")
        hist = fit(m, make, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed, log_every=max(1, a.steps // 10))
        for h in hist:
            print(f"step {h['step']:5d}  loss {h['loss']:.4f}  tok_acc {h['tok_acc']:.3f}  exact {h['exact']:.3f}")
        if a.save:
            m.save(a.save)
            print("saved", a.save)
    else:
        m = Mamba.load(a.checkpoint)
        if m.cfg["vocab"] != vocab:
            print(f"checkpoint vocab {m.cfg['vocab']} does not match this task's {vocab}", file=sys.stderr)
            return 2
        toks, tgts = make(a.n, np.random.default_rng(12345))
        tok, exact = score(m.forward(toks), tgts)
        print(f"{a.task}: token accuracy {tok:.3f}, exact {exact:.3f} on {a.n} fresh samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
