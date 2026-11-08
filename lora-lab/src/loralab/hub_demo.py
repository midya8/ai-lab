"""Adapter hub demo: one frozen char-GPT, two domain adapters, swap / blend / route / compose.

    PYTHONPATH=src python -m loralab.hub_demo

Pretrain on Shakespeare (domain A), train a 'tech' adapter (B) and a 'kitchen' adapter (C)
independently from the same frozen base, then score every configuration on held-out text
from all three domains. All domain-B/C text is synthetic template output.
"""
from __future__ import annotations

import argparse

import numpy as np

from . import gpt_lora  # noqa: F401  (sets up the tinygpt import path)
from .domain import generate_domain_text
from .gpt_demo import CORPUS, batch, split
from .hub import AdapterHub, BASE, train_adapter
from tinygpt import GPT, GPTConfig, Adam, CharTokenizer


def run(rank=4, pre_steps=250, ad_steps=150, block=32, embd=48, layers=2, heads=4, seed=0,
        route_samples=40, prompt_len=48):
    text_a = open(CORPUS, encoding="utf-8").read()
    text_b = generate_domain_text(150, seed=7, domain="tech")
    text_c = generate_domain_text(150, seed=8, domain="kitchen")
    tok = CharTokenizer(text_a + text_b + text_c)
    parts = {d: split(t, tok, block) for d, t in zip("ABC", (text_a, text_b, text_c))}
    rng = np.random.default_rng(seed)

    base = GPT(GPTConfig(tok.vocab_size, block, layers, heads, embd), seed=seed)
    opt = Adam(base.params, lr=3e-3)
    for _ in range(pre_steps):
        x, y = batch(parts["A"][0], 16, block, rng)
        opt.step(base.loss_and_grads(x, y)[1])
    snapshot = {k: v.copy() for k, v in base.params.items()}

    hub = AdapterHub(base, ("qkv", "proj"), r=rank)
    train_adapter(hub.add("tech", seed=1), parts["B"][0], ad_steps, block, seed=1)
    train_adapter(hub.add("kitchen", seed=2), parts["C"][0], ad_steps, block, seed=2)

    configs = {BASE: None, "tech": "tech", "kitchen": "kitchen", "blend 50/50": {"tech": .5, "kitchen": .5}}
    nll = {}
    for label, spec in configs.items():
        nll[label] = {d: hub.evaluate_loss(parts[d][1], block, spec) for d in "ABC"}

    mix, err = hub.compose({"tech": .5, "kitchen": .5}, "mix", rank=2 * rank)
    nll["composed (SVD)"] = {d: hub.evaluate_loss(parts[d][1], block, "mix") for d in "ABC"}

    # routing: label-free choice of adapter from a short held-out snippet
    r = np.random.default_rng(99)
    truth = {"B": "tech", "C": "kitchen"}
    right = total = 0
    for d, want in truth.items():
        data = parts[d][1]
        for _ in range(route_samples):
            s = int(r.integers(0, len(data) - prompt_len - 1))
            best, _ = hub.route(data[s:s + prompt_len + 1], candidates=["tech", "kitchen"])
            right += best == want
            total += 1
    return {
        "nll": nll, "route_accuracy": right / total, "compose_rel_error": err,
        "adapter_params": hub.adapters["tech"].n_trainable(), "full_params": base.num_params(),
        "base_frozen": all(np.array_equal(snapshot[k], base.params[k]) for k in snapshot),
        "hub": hub,
    }


def format_report(o):
    lines = ["config           NLL A (Shakespeare)  NLL B (tech)  NLL C (kitchen)"]
    for name, row in o["nll"].items():
        lines.append(f"{name:<16} {row['A']:>19.3f}  {row['B']:>12.3f}  {row['C']:>15.3f}")
    lines.append(f"routing accuracy from a 48-char snippet (tech vs kitchen): {o['route_accuracy']:.3f}")
    lines.append(f"compose rank 2r vs exact blend: relative error {o['compose_rel_error']:.2e}")
    lines.append(f"each adapter: {o['adapter_params']:,} params ({100 * o['adapter_params'] / o['full_params']:.1f}% of {o['full_params']:,}); base frozen: {o['base_frozen']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Multi-adapter hub demo")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--pre-steps", type=int, default=250)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--save", help="directory to save the trained hub into")
    a = ap.parse_args(argv)
    o = run(a.rank, a.pre_steps, a.steps)
    print(format_report(o))
    if a.save:
        o["hub"].save(a.save)
        print(f"saved hub to {a.save}")


if __name__ == "__main__":
    main()
