"""Memory-vs-quality frontier for block size x double quantization.

    PYTHONPATH=src python -m loralab.dq_frontier [--seeds 3] [--schemes nf4,int4] [--blocks 16,32,64,128,256]

Small blocks track local weight scale better (lower error) but pay for one fp32 scale per block.
Double quantization shrinks that overhead (0.5 -> ~0.13 bits/weight at block 64), so the question is
whether the same bit budget buys a better model with a smaller block + double quantization than with
a bigger block and plain fp32 scales. For every (scheme, block, double_quant) this reports stored
bits per weight, relative weight error, and held-out NLL of the *quantized base alone* on domain A
(no adapters, so it isolates the quantizer). Pareto-optimal rows are marked.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import gpt_lora  # noqa: F401  (sets up the tinygpt import path)
from .gpt_demo import CORPUS, batch, split
from .gpt_lora import GPTLoRA
from .qlora_demo import clone
from .quant import QuantizedBase
from tinygpt import GPT, Adam, GPTConfig, CharTokenizer


def pareto(points):
    """Indices of points not dominated on (bits, error): no other point is <= on both and < on one."""
    keep = []
    for i, (b, e) in enumerate(points):
        if not any((b2 <= b and e2 <= e) and (b2 < b or e2 < e) for j, (b2, e2) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def pretrain(seed, block_size, steps, embd=48, layers=2, heads=4):
    text = open(CORPUS, encoding="utf-8").read()
    tok = CharTokenizer(text)
    tr, va = split(text, tok, block_size)
    rng = np.random.default_rng(seed)
    base = GPT(GPTConfig(tok.vocab_size, block_size, layers, heads, embd), seed=seed)
    opt = Adam(base.params, lr=3e-3)
    for _ in range(steps):
        x, y = batch(tr, 16, block_size, rng)
        opt.step(base.loss_and_grads(x, y)[1])
    return base, va


def run(seeds=3, schemes=("nf4", "int4"), blocks=(16, 32, 64, 128, 256), pre_steps=250, block_size=32, dq_block=256):
    acc = {}
    fp_nll = []
    for seed in range(seeds):
        base, va = pretrain(seed, block_size, pre_steps)
        fp_nll.append(GPTLoRA(clone(base), ("qkv",), r=1, seed=0).evaluate_loss(va, block_size, enabled=False))
        for scheme in schemes:
            for blk in blocks:
                for dq in (False, True):
                    m = clone(base)
                    qb = QuantizedBase(m, scheme, block=blk, double_quant=dq, dq_block=dq_block)
                    nll = GPTLoRA(m, ("qkv",), r=1, seed=0).evaluate_loss(va, block_size, enabled=False)
                    a = acc.setdefault((scheme, blk, dq), {"bits": qb.bits_per_weight(), "err": [], "nll": []})
                    a["err"].append(qb.relative_error())
                    a["nll"].append(nll)
    rows = [{"scheme": k[0], "block": k[1], "double_quant": k[2], "bits": v["bits"],
             "err": float(np.mean(v["err"])), "nll": float(np.mean(v["nll"])),
             "nll_std": float(np.std(v["nll"]))} for k, v in acc.items()]
    for scheme in schemes:
        idx = [i for i, r in enumerate(rows) if r["scheme"] == scheme]
        for j in pareto([(rows[i]["bits"], rows[i]["err"]) for i in idx]):
            rows[idx[j]]["pareto"] = True
    return rows, float(np.mean(fp_nll))


def format_report(rows, fp_nll):
    lines = [f"fp32 base NLL on A: {fp_nll:.3f}",
             "scheme  block  double-quant  bits/weight  weight err  NLL A (mean ± sd)  pareto"]
    for r in sorted(rows, key=lambda r: (r["scheme"], -r["bits"])):
        lines.append(f"{r['scheme']:<6}  {r['block']:>5}  {'yes' if r['double_quant'] else 'no':<12}  {r['bits']:>11.3f}  "
                     f"{r['err']:>10.4f}  {r['nll']:>9.3f} ± {r['nll_std']:.3f}   {'*' if r.get('pareto') else ''}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="block size x double quantization frontier")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--schemes", default="nf4,int4")
    ap.add_argument("--blocks", default="16,32,64,128,256")
    ap.add_argument("--pre-steps", type=int, default=250)
    a = ap.parse_args(argv)
    print(format_report(*run(a.seeds, tuple(a.schemes.split(",")), tuple(int(b) for b in a.blocks.split(",")), a.pre_steps)))


if __name__ == "__main__":
    main()
