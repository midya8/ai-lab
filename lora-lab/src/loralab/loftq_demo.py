"""LoftQ demo: does a quantization-aware adapter init beat QLoRA's zero init?

    PYTHONPATH=src python -m loralab.loftq_demo [--rank 4] [--iters 5] [--schemes nf4,int4]

Pretrain a char GPT on Shakespeare (domain A) in fp32. For each quantization scheme, build three
starting points over the *same* quantized-model recipe, adapters on all four linear weights:

* ``qlora``: quantize W, adapters start at zero (B = 0),
* ``svd``:   quantize W, adapters start as the rank-r SVD of the quantization error (LoftQ, 1 iteration),
* ``loftq``: alternate quantize / SVD for ``--iters`` iterations.

Reported: weight error of Q + adapter vs fp32 W, NLL on A *at step 0* (nothing trained yet), then NLL on
the synthetic tech domain B after a short adapter fine-tune, and NLL on A after that fine-tune.
One seed; the model has 62k parameters; B is template text. Read differences of a few hundredths as noise.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import gpt_lora  # noqa: F401  (sets up the tinygpt import path)
from .domain import generate_domain_text
from .gpt_demo import CORPUS, batch, split
from .gpt_lora import GPTLoRA
from .loftq import loftq_init
from .qlora_demo import clone, train
from .quant import QuantizedBase
from tinygpt import GPT, Adam, GPTConfig, CharTokenizer

TARGETS = ("qkv", "proj", "fc", "out")


def run(rank=4, iters=5, block_size=64, pre_steps=250, ad_steps=150, block=32, embd=48, layers=2, heads=4,
        seed=0, schemes=("nf4", "int4")):
    text_a = open(CORPUS, encoding="utf-8").read()
    text_b = generate_domain_text(150, seed=7)
    tok = CharTokenizer(text_a + text_b)
    a_tr, a_va = split(text_a, tok, block)
    b_tr, b_va = split(text_b, tok, block)
    rng = np.random.default_rng(seed)
    base = GPT(GPTConfig(tok.vocab_size, block, layers, heads, embd), seed=seed)
    opt = Adam(base.params, lr=3e-3)
    for _ in range(pre_steps):
        x, y = batch(a_tr, 16, block, rng)
        opt.step(base.loss_and_grads(x, y)[1])
    fp_a = GPTLoRA(clone(base), TARGETS, r=rank, seed=seed).evaluate_loss(a_va, block, enabled=False)

    rows = []
    for scheme in schemes:
        for init in ("qlora", "svd", "loftq"):
            m = clone(base)
            lo = GPTLoRA(m, TARGETS, r=rank, seed=seed)
            if init == "qlora":
                qb = QuantizedBase(m, scheme, block=block_size)
                werr = qb.relative_error()
            else:
                res = loftq_init(lo, scheme, block_size, iters=1 if init == "svd" else iters)
                qb, werr = res.qbase, res.final_error
            row = {"scheme": scheme, "init": init, "weight_err": werr,
                   "A_step0": lo.evaluate_loss(a_va, block), "B_step0": lo.evaluate_loss(b_va, block)}
            train(lo, b_tr, ad_steps, block, seed=1)
            row["B_trained"] = lo.evaluate_loss(b_va, block)
            row["A_after_B"] = lo.evaluate_loss(a_va, block)
            row["base_intact"] = qb.verify_frozen()
            rows.append(row)
    return fp_a, rows


def format_report(fp_a, rows):
    lines = [f"fp32 pretrained NLL on A: {fp_a:.3f}",
             "scheme init    weight err  NLL A (step 0)  NLL B (step 0)  NLL B (trained)  NLL A (after B)"]
    for r in rows:
        lines.append(f"{r['scheme']:<6} {r['init']:<7} {r['weight_err']:>10.4f}  {r['A_step0']:>14.3f}  "
                     f"{r['B_step0']:>14.3f}  {r['B_trained']:>15.3f}  {r['A_after_B']:>15.3f}")
    lines.append(f"quantized bases untouched by training: {all(r['base_intact'] for r in rows)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LoftQ vs QLoRA adapter initialisation")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--block", type=int, default=64, help="quantization block size")
    ap.add_argument("--schemes", default="nf4,int4")
    ap.add_argument("--pre-steps", type=int, default=250)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args(argv)
    print(format_report(*run(a.rank, a.iters, a.block, a.pre_steps, a.steps, schemes=tuple(a.schemes.split(",")))))


if __name__ == "__main__":
    main()
