"""Train MiniGPT (from-scratch autodiff) on tiny-transformer's corpus.
Caveat: the corpus is ~2.6KB and repetitive, so val loss is optimistic (leakage)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data, train, generate

text = open(os.path.join(os.path.dirname(__file__), "..", "..", "tiny-transformer", "data", "corpus.txt")).read()
chars, tr, va = make_data(text)
m = MiniGPT(len(chars), d=32, heads=4, layers=2, ctx=32, seed=0)
t = time.time()
hist = train(m, tr, va, 32, steps=int(sys.argv[1]) if len(sys.argv) > 1 else 400, bs=16, lr=3e-3, log_every=50)
print("vocab", len(chars), "uniform loss", round(float(np.log(len(chars))), 3))
for s, tl, vl in hist:
    print(f"step {s:4d} train {tl:.3f} val {vl:.3f}")
print("time", round(time.time() - t, 1), "s")
print(generate(m, chars, tr[:8], 150, 32))
