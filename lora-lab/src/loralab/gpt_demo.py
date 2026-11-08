"""Adapt a Shakespeare-pretrained tiny GPT to a technical-prose domain with LoRA on attention.

    PYTHONPATH=src python -m loralab.gpt_demo --rank 4
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import gpt_lora  # noqa: F401  (sets up the tinygpt import path)
from .domain import generate_domain_text
from .gpt_lora import GPTLoRA
from tinygpt import GPT, GPTConfig, Adam, CharTokenizer

CORPUS = os.path.join(os.path.dirname(__file__), "..", "..", "..", "tiny-transformer", "data", "corpus.txt")


def batch(data, B, T, rng):
    ix = rng.integers(0, len(data) - T - 1, B)
    return np.stack([data[i:i + T] for i in ix]), np.stack([data[i + 1:i + T + 1] for i in ix])


def split(text, tok, block):
    d = np.array(tok.encode(text))
    n = max(int(len(d) * 0.15), block + 2)
    return d[:-n], d[-n:]


def run(rank=4, targets=("qkv", "proj"), pre_steps=250, ad_steps=150, block=32, embd=48, layers=2, heads=4,
        lr_pre=3e-3, lr_ad=1e-2, seed=0, corpus=None):
    """Returns a dict of held-out NLL (nats/char) and parameter counts."""
    text_a = open(corpus or CORPUS, encoding="utf-8").read()
    text_b = generate_domain_text(150, seed=7)
    tok = CharTokenizer(text_a + text_b)  # shared vocab so both domains fit one embedding
    a_tr, a_va = split(text_a, tok, block)
    b_tr, b_va = split(text_b, tok, block)
    rng = np.random.default_rng(seed)

    base = GPT(GPTConfig(tok.vocab_size, block, layers, heads, embd), seed=seed)
    opt = Adam(base.params, lr=lr_pre)
    for _ in range(pre_steps):
        x, y = batch(a_tr, 16, block, rng)
        opt.step(base.loss_and_grads(x, y)[1])

    out = {"full_params": base.num_params()}
    snapshot = {k: v.copy() for k, v in base.params.items()}

    lora = GPTLoRA(base, targets, r=rank, seed=seed)
    out["base_A"], out["base_B"] = (lora.evaluate_loss(a_va, block, enabled=False), lora.evaluate_loss(b_va, block, enabled=False))
    ad_opt = Adam(lora.params(), lr=lr_ad)
    for _ in range(ad_steps):
        x, y = batch(b_tr, 16, block, rng)
        ad_opt.step(lora.loss_and_grads(x, y)[1])
    out["lora_A"], out["lora_B"] = (lora.evaluate_loss(a_va, block), lora.evaluate_loss(b_va, block))
    out["lora_params"] = lora.n_trainable()
    out["base_frozen"] = all(np.array_equal(snapshot[k], base.params[k]) for k in snapshot)
    lora.merge()
    out["merged_B"] = lora.evaluate_loss(b_va, block)
    lora.unmerge()

    # full fine-tune reference from the same pretrained weights
    for k, v in snapshot.items():
        base.params[k][...] = v
    fopt = Adam(base.params, lr=lr_ad / 5)
    for _ in range(ad_steps):
        x, y = batch(b_tr, 16, block, rng)
        fopt.step(base.loss_and_grads(x, y)[1])
    out["full_A"], out["full_B"] = (_plain(base, a_va, block), _plain(base, b_va, block))
    return out


def _plain(model, data, block, batches=20, batch_size=16):
    r = np.random.default_rng(0)
    return float(np.mean([model.forward(*batch(data, batch_size, block, r))[1] for _ in range(batches)]))


def format_report(o):
    pct = 100 * o["lora_params"] / o["full_params"]
    rows = [("pretrained (no adapter)", o["base_A"], o["base_B"], 0),
            ("LoRA (frozen base)", o["lora_A"], o["lora_B"], o["lora_params"]),
            ("full fine-tune", o["full_A"], o["full_B"], o["full_params"])]
    lines = ["method                    NLL domain A (Shakespeare)  NLL domain B (tech)  trainable"]
    for name, a, b, n in rows:
        lines.append(f"{name:<25} {a:>26.3f}  {b:>19.3f}  {n:>9,}")
    lines.append(f"LoRA trains {pct:.1f}% of weights; merged-model NLL on B {o['merged_B']:.3f}; base frozen: {o['base_frozen']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="LoRA on tiny-GPT attention")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--targets", default="qkv,proj", help="comma list of qkv,proj,fc,out")
    ap.add_argument("--pre-steps", type=int, default=250)
    ap.add_argument("--steps", type=int, default=150)
    a = ap.parse_args(argv)
    print(format_report(run(a.rank, tuple(a.targets.split(",")), a.pre_steps, a.steps)))


if __name__ == "__main__":
    main()
