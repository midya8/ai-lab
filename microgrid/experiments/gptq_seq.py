"""GPTQ-style vs round-to-nearest at 2-4 bits (per-channel), held-out stdlib source, 3 seeds.
Calibration windows come from the TRAIN split only; evaluation is on the disjoint held-out files."""
import os, sys, glob, sysconfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data_split, train
from microgrid.kvcache import cached_nll
from microgrid.wquant import quantize_model
from microgrid.gptq import gptq_model
from train_heldout import load

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    chars, tr, va = make_data_split(load(files[0::2], 60000), load(files[1::2], 12000))
    va = va[:3000]
    bits_list = (4, 3, 2)
    res = {(b, k): [] for b in bits_list for k in ("rtn", "gptq", "seq")}
    for s in (0, 1, 2):
        m = MiniGPT(len(chars), d=48, heads=4, layers=2, ctx=48, seed=s)
        train(m, tr, va, 48, steps=steps, bs=16, lr=3e-3, log_every=steps, clip=1.0, warmup=30,
              schedule=True, seed=s)
        base, _ = cached_nll(m, va)
        rng = np.random.default_rng(100 + s)
        st = rng.integers(0, len(tr) - 49, 64)
        win = np.stack([tr[i:i + 48] for i in st])
        print(f"seed {s}: fp64 nll {base:.4f}")
        for b in bits_list:
            r, _ = cached_nll(quantize_model(m, b, True), va)
            g, _ = cached_nll(gptq_model(m, win, b), va)
            q, _ = cached_nll(gptq_model(m, win, b, sequential=True), va)
            res[(b, "seq")].append(q - base)
            res[(b, "rtn")].append(r - base); res[(b, "gptq")].append(g - base)
            print(f"  {b}-bit: rtn {r-base:+.4f}  gptq {g-base:+.4f}  seq {q-base:+.4f}")
    print("mean delta nats over 3 seeds (min..max):")
    for (b, k), d in res.items():
        print(f"  {b}-bit {k:4s}: {np.mean(d):+.4f} ({min(d):+.4f}..{max(d):+.4f})")
