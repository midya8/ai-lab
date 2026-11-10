"""Train MiniGPT on real, non-repetitive text with a leak-free split.
Corpus: Python stdlib source files (ASCII only). Train and validation are DIFFERENT files,
so val loss measures generalisation to unseen code, unlike train_corpus.py."""
import os, sys, time, glob, sysconfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data_split, train, generate


def load(files, limit):
    out = []
    for f in files:
        t = open(f, encoding="utf-8", errors="ignore").read()
        if t.isascii():
            out.append(t)
        if sum(map(len, out)) >= limit:
            break
    return "".join(out)[:limit]


if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    # alternate files: even -> train, odd -> held-out (disjoint)
    tr_text = load(files[0::2], 60000)
    va_text = load(files[1::2], 12000)
    chars, tr, va = make_data_split(tr_text, va_text)
    m = MiniGPT(len(chars), d=48, heads=4, layers=2, ctx=48, seed=0)
    t = time.time()
    hist = train(m, tr, va, 48, steps=steps, bs=16, lr=3e-3, log_every=max(1, steps // 8),
                 clip=1.0, warmup=30, schedule=True)
    print("train chars", len(tr), "held-out chars", len(va), "vocab", len(chars),
          "uniform loss", round(float(np.log(len(chars))), 3))
    for s, tl, vl in hist:
        print(f"step {s:4d} train {tl:.3f} heldout-val {vl:.3f}")
    print("time", round(time.time() - t, 1), "s")
    print(generate(m, chars, tr[:8], 200, 48))
