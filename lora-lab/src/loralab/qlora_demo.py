"""QLoRA-style demo: LoRA on a frozen, quantized base.

    PYTHONPATH=src python -m loralab.qlora_demo [--rank 4] [--block 64]

Pretrain a char GPT on Shakespeare (domain A) in float32, then for each base precision
(fp32, int8, nf4, int4):

1. quantize the linear weights and measure how much the *base* model degrades on A,
2. train rank-r attention adapters on the synthetic tech domain (B) over that frozen base,
3. train adapters on A itself to see how much of the quantization damage they win back.

Domain B text is synthetic template output; the model is 62k parameters, so quantization hurts
far more than it would on a large model whose weights are more redundant.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import gpt_lora  # noqa: F401  (sets up the tinygpt import path)
from .domain import generate_domain_text
from .gpt_demo import CORPUS, batch, split
from .gpt_lora import GPTLoRA
from .quant import QuantizedBase
from tinygpt import GPT, Adam, GPTConfig, CharTokenizer


def clone(model):
    m = GPT(model.cfg, seed=0, dtype=next(iter(model.params.values())).dtype)
    for k, v in model.params.items():
        m.params[k] = v.copy()
    return m


def train(lora, data, steps, block, lr=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    opt = Adam(lora.params(), lr=lr)
    for _ in range(steps):
        x, y = batch(data, 16, block, rng)
        opt.step(lora.loss_and_grads(x, y)[1])


def run(rank=4, block_size=64, pre_steps=250, ad_steps=150, block=32, embd=48, layers=2, heads=4, seed=0,
        schemes=("fp32", "int8", "nf4", "int4")):
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

    rows = []
    for scheme in schemes:
        m = clone(base)
        qb = None if scheme == "fp32" else QuantizedBase(m, scheme, block=block_size)
        lo = GPTLoRA(m, ("qkv", "proj"), r=rank, seed=seed)
        row = {"scheme": scheme,
               "model_bytes": qb.model_bytes() if qb else sum(v.nbytes for v in m.params.values()),
               "weight_err": qb.relative_error() if qb else 0.0,
               "base_A": lo.evaluate_loss(a_va, block, enabled=False),
               "base_B": lo.evaluate_loss(b_va, block, enabled=False)}
        train(lo, b_tr, ad_steps, block, seed=1)
        row["qlora_B"] = lo.evaluate_loss(b_va, block)
        row["adapter_params"] = lo.n_trainable()
        row["base_intact"] = qb.verify_frozen() if qb else True

        # recovery: fresh adapters on a fresh copy, trained on A itself
        m2 = clone(base)
        if scheme != "fp32":
            QuantizedBase(m2, scheme, block=block_size)
        rec = GPTLoRA(m2, ("qkv", "proj"), r=rank, seed=seed)
        train(rec, a_tr, ad_steps, block, seed=2)
        row["recovered_A"] = rec.evaluate_loss(a_va, block)
        rows.append(row)
    return rows


def format_report(rows):
    ref = next((r for r in rows if r["scheme"] == "fp32"), rows[0])
    lines = ["base   model KiB  weight err  NLL A (base)  NLL B (base)  NLL B (+LoRA)  NLL A (+LoRA on A)"]
    for r in rows:
        lines.append(f"{r['scheme']:<6} {r['model_bytes'] / 1024:>9.1f}  {r['weight_err']:>10.4f}  {r['base_A']:>12.3f}  "
                     f"{r['base_B']:>12.3f}  {r['qlora_B']:>13.3f}  {r['recovered_A']:>18.3f}")
    lines.append(f"adapters: {ref['adapter_params']:,} trainable params, fp32, identical across rows; "
                 f"quantized bases untouched by training: {all(r['base_intact'] for r in rows)}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="QLoRA-style LoRA on a quantized frozen base")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--block", type=int, default=64, help="quantization block size")
    ap.add_argument("--schemes", default="fp32,int8,nf4,int4")
    ap.add_argument("--pre-steps", type=int, default=250)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args(argv)
    print(format_report(run(a.rank, a.block, a.pre_steps, a.steps, schemes=tuple(a.schemes.split(",")))))


if __name__ == "__main__":
    main()
