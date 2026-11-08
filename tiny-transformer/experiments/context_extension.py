"""Context extension for a RoPE model: extrapolate vs position interpolation vs NTK scaling.

    PYTHONPATH=src python experiments/context_extension.py [--seeds 0 1 2] [--steps 300] [--ft-steps 60]

One RoPE model per seed is trained at ``--block`` tokens. For every longer target length it is
then stretched four ways, all sharing the same trained weights:

* ``none``    plain extrapolation (longer window, original angles): the baseline
* ``linear``  position interpolation (positions / (target / block))
* ``ntk``     NTK-aware base rescaling

each scored zero-shot and again after a short fine-tune at the target length (same data, same
steps, same seed for every method). Every cell is the sliding-window loss (nats/char) over the
*same* held-out tokens, each scored with up to `target` tokens of context, so numbers compare
across lengths. The last column re-scores the fine-tuned model at the original block length
to show what the stretch cost short contexts (same angle schedule, shorter window). Results are averaged over seeds; the spread
between methods is often within seed noise at this scale, and the README reports it as such.

Caveat: the bundled corpus is small and repetitive; every 20-gram of its held-out tail also occurs in
the training portion. So absolute losses are low and the *fine-tuned* columns largely measure how
well the model refits memorised text, not generalisation. The zero-shot columns (how badly each
scaling breaks an unmodified model) and the "@ block" retention column are the informative ones.
"""
import argparse
import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tinygpt import GPT, GPTConfig, Adam, CharTokenizer  # noqa: E402
from tinygpt.checkpoint import evaluate_sliding  # noqa: E402
from tinygpt.extend import extend_context, finetune_extended, with_window  # noqa: E402
from tinygpt.train import get_batch  # noqa: E402

HERE = os.path.dirname(__file__)
MODES = ("none", "linear", "ntk")


def train_base(data, vocab, args, seed):
    m = GPT(GPTConfig(vocab, args.block, args.layers, args.heads, args.embd, rope=True), seed=seed)
    opt = Adam(m.params, lr=args.lr)
    rng = np.random.default_rng(seed)
    for _ in range(args.steps):
        x, y = get_batch(data, args.batch, args.block, rng)
        _, g = m.loss_and_grads(x, y)
        opt.step(g)
    return m


def sliding(model, val, stride):
    return evaluate_sliding(model, val, stride=min(stride, model.cfg.block_size))["loss"]


def run_seed(train_d, val, vocab, args, seed):
    base = train_base(train_d, vocab, args, seed)
    out = {"in_window": sliding(base, val, args.stride), "cells": {}}
    for L in args.lengths:
        for mode in MODES:
            frozen = copy.deepcopy(base.params)
            view = extend_context(base, L, mode)
            zero = sliding(view, val, args.stride)
            finetune_extended(view, train_d, args.ft_steps, batch=args.ft_batch, lr=args.ft_lr, seed=seed)
            tuned = sliding(view, val, args.stride)
            short = sliding(with_window(view, args.block), val, args.stride)
            out["cells"][(L, mode)] = (zero, tuned, short)
            for k in base.params:  # restore the shared weights for the next method
                base.params[k][...] = frozen[k]
    return out


def run(args):
    text = open(os.path.join(HERE, "..", "data", "corpus.txt"), encoding="utf-8").read()
    tok = CharTokenizer(text)
    data = np.array(tok.encode(text))
    cut = int(len(data) * 0.85)
    train_d, val = data[:cut], data[cut:]
    per_seed = [run_seed(train_d, val, tok.vocab_size, args, s) for s in args.seeds]
    res = {"in_window": float(np.mean([r["in_window"] for r in per_seed])), "cells": {}, "seeds": len(per_seed)}
    for key in per_seed[0]["cells"]:
        res["cells"][key] = tuple(float(np.mean([r["cells"][key][i] for r in per_seed])) for i in range(3))
    return res


def fmt(res, lengths):
    out = [f"trained-window baseline (sliding, within block): {res['in_window']:.3f}  (mean of {res['seeds']} seeds)", "",
           "| target len | method | zero-shot | after fine-tune | fine-tuned @ block |", "|---|---|---|---|---|"]
    for L in lengths:
        for mode in MODES:
            z, t, s = res["cells"][(L, mode)]
            out.append(f"| {L} | {mode} | {z:.3f} | {t:.3f} | {s:.3f} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lengths", type=int, nargs="+", default=[64, 96])
    ap.add_argument("--ft-steps", type=int, default=60)
    ap.add_argument("--ft-batch", type=int, default=8)
    ap.add_argument("--ft-lr", type=float, default=1e-3)
    ap.add_argument("--stride", type=int, default=8)
    args = ap.parse_args(argv)
    print(f"RoPE trained at block={args.block} for {args.steps} steps; fine-tune {args.ft_steps} steps")
    print(fmt(run(args), args.lengths))


if __name__ == "__main__":
    main()
