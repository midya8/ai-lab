"""Perplexity cost of weight-only quantisation on held-out stdlib source (same split as kv_perplexity.py)."""
import os, sys, glob, sysconfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data_split, train
from microgrid.kvcache import cached_nll
from microgrid.wquant import quantize_model, weight_bytes
from train_heldout import load

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    chars, tr, va = make_data_split(load(files[0::2], 60000), load(files[1::2], 12000))
    va = va[:3000]
    cfgs = [(8, True), (6, True), (4, True), (4, False), (3, True), (2, True)]
    deltas = {c: [] for c in cfgs}
    for s in (0, 1, 2):
        m = MiniGPT(len(chars), d=48, heads=4, layers=2, ctx=48, seed=s)
        train(m, tr, va, 48, steps=steps, bs=16, lr=3e-3, log_every=steps, clip=1.0, warmup=30,
              schedule=True, seed=s)
        base, _ = cached_nll(m, va)
        print(f"seed {s}: fp64 nll {base:.4f} ({weight_bytes(m)} B)")
        for c in cfgs:
            q, _ = cached_nll(quantize_model(m, c[0], c[1]), va)
            deltas[c].append(q - base)
            print(f"  {c[0]}-bit {'per-channel' if c[1] else 'per-tensor '}: nll {q:.4f} delta {q-base:+.4f}"
                  f" | {weight_bytes(m, c[0], c[1])} B")
    print("mean delta nats over 3 seeds:")
    for c, d in deltas.items():
        print(f"  {c[0]}-bit {'per-channel' if c[1] else 'per-tensor '}: {np.mean(d):+.4f} (min {min(d):+.4f}, max {max(d):+.4f})")
