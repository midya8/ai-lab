"""Train a char-level GPT:  python -m tinygpt.train --data data/shakespeare_snippet.txt"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .model import GPT, GPTConfig
from .optim import Adam
from .tokenizer import CharTokenizer
from .bpe import BPETokenizer
from .schedule import warmup_cosine
from .accum import accumulate_grads
from .clip import clip_grad_norm
from .earlystop import EarlyStopper
from .lrfind import find_lr
from .checkpoint import save_checkpoint, evaluate


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
    ap.add_argument("--micro-batch", type=int, default=0, help="split each batch into micro-batches (gradient accumulation)")
    ap.add_argument("--clip", type=float, default=0.0, help="clip global gradient norm to this value (0 = off)")
    ap.add_argument("--label-smoothing", type=float, default=0.0, help="label smoothing epsilon in [0,1) for the training loss")
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout prob on embeddings, attention and residuals (train only)")
    ap.add_argument("--tie-weights", action="store_true", help="share token embedding with the output head")
    ap.add_argument("--patience", type=int, default=0, help="early stopping: evals without val improvement before stopping (0 = off)")
    ap.add_argument("--eval-every", type=int, default=50, help="validation interval in steps when --patience is set")
    ap.add_argument("--min-delta", type=float, default=0.0, help="minimum val-loss improvement that counts")
    ap.add_argument("--find-lr", action="store_true", help="run an LR range test, print a suggested LR and exit (weights untouched)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--vocab", type=int, default=300, help="BPE vocab size")
    ap.add_argument("--val-frac", type=float, default=0.1, help="held-out tail fraction for perplexity")
    ap.add_argument("--out", default="tinygpt.npz")
    ap.add_argument("--sample", type=int, default=200)
    a = ap.parse_args(argv)

    text = open(a.data, encoding="utf-8").read()
    tok = CharTokenizer(text) if a.tokenizer == "char" else BPETokenizer.train(text, a.vocab)
    data = np.array(tok.encode(text))
    n_val = int(len(data) * a.val_frac)
    val = data[len(data) - n_val:] if n_val > a.block + 1 else None
    if val is not None:
        data = data[:len(data) - n_val]
    model = GPT(GPTConfig(tok.vocab_size, a.block, a.layers, a.heads, a.embd, a.dropout, a.tie_weights))
    opt = Adam(model.params, lr=a.lr, weight_decay=a.weight_decay)
    if a.find_lr:
        r = find_lr(model, data, a.batch, a.block, steps=min(a.steps, 100), get_batch=get_batch)
        for lr, l in list(zip(r["lrs"], r["losses"]))[::max(1, len(r["lrs"]) // 15)]:
            print(f"lr {lr:.2e}  loss {l:.3f}")
        print(f"suggested --lr {r['suggested']:.2e}" + (" (sweep stopped at divergence)" if r["stopped_early"] else ""))
        return
    rng = np.random.default_rng(0)
    model.label_smoothing = a.label_smoothing
    drop_rng = np.random.default_rng(1) if a.dropout else None
    print(f"vocab={tok.vocab_size} params={model.num_params():,} chars={len(text):,}")
    stopper = EarlyStopper(a.patience, a.min_delta) if a.patience and val is not None else None
    if a.patience and val is None:
        print("warning: --patience ignored (validation split too small)")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        opt.lr = warmup_cosine(step, a.steps, a.lr, a.warmup)
        x, y = get_batch(data, a.batch, a.block, rng)
        if a.micro_batch:
            loss, grads = accumulate_grads(model, x, y, a.micro_batch, rng=drop_rng)
        else:
            loss, grads = model.loss_and_grads(x, y, rng=drop_rng)
        gnorm = None
        if a.clip:
            grads, gnorm = clip_grad_norm(grads, a.clip)
        opt.step(grads)
        if step % 50 == 0 or step == 1:
            extra = f" gnorm {gnorm:.2f}" if gnorm is not None else ""
            print(f"step {step:4d} loss {loss:.3f}{extra} ({time.time() - t0:.0f}s)")
        if stopper is not None and step % a.eval_every == 0:
            vl = evaluate(model, val)["loss"]
            print(f"step {step:4d} val loss {vl:.3f} (best {min(vl, stopper.best):.3f})")
            if stopper.update(vl, model.params, step):
                print(f"early stop at step {step}; restoring best weights from step {stopper.best_step}")
                break
    if stopper is not None:
        stopper.restore(model.params)
    save_checkpoint(a.out, model, tok)
    if val is not None:
        r = evaluate(model, val)
        print(f"held-out loss {r['loss']:.3f} perplexity {r['perplexity']:.2f} ({r['tokens']} tokens)")
    seed = tok.encode(text[:8])
    print("--- sample ---")
    print(tok.decode(model.generate(seed, a.sample, temperature=0.8, top_k=10)))


if __name__ == "__main__":
    main()
