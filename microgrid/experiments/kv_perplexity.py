"""Perplexity of exact vs int8 KV cache on held-out stdlib source (same split as train_heldout.py)."""
import os, sys, glob, sysconfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data_split, train
from microgrid.kvcache import cached_nll, KVCache, QuantKVCache
from train_heldout import load

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    chars, tr, va = make_data_split(load(files[0::2], 60000), load(files[1::2], 12000))
    va = va[:3000]
    for s in (0, 1, 2):
        m = MiniGPT(len(chars), d=48, heads=4, layers=2, ctx=48, seed=s)
        train(m, tr, va, 48, steps=steps, bs=16, lr=3e-3, log_every=steps, clip=1.0, warmup=30,
              schedule=True, seed=s)
        e, n = cached_nll(m, va, KVCache)
        q, _ = cached_nll(m, va, QuantKVCache)
        print(f"seed {s}: n={n} exact nll {e:.4f} (ppl {np.exp(e):.3f}) | int8 nll {q:.4f} "
              f"(ppl {np.exp(q):.3f}) | delta {q-e:+.5f} nats")
