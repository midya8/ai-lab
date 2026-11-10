"""Multi-seed held-out run (same leak-free split as train_heldout.py) with a weight-decay ablation.
Seeds change model init AND batch sampling (via train's seed if supported). Reports mean +- std."""
import os, sys, glob, sysconfig, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from microgrid import MiniGPT
from microgrid.lm import make_data_split, train
from train_heldout import load

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    seeds = [0, 1, 2]
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    chars, tr, va = make_data_split(load(files[0::2], 60000), load(files[1::2], 12000))
    has_seed = "seed" in inspect.signature(train).parameters
    for wd in (0.0, 0.1):
        finals = []
        for s in seeds:
            m = MiniGPT(len(chars), d=48, heads=4, layers=2, ctx=48, seed=s)
            kw = dict(seed=s) if has_seed else {}
            h = train(m, tr, va, 48, steps=steps, bs=16, lr=3e-3, log_every=steps,
                      clip=1.0, warmup=30, schedule=True, weight_decay=wd, **kw)
            finals.append(h[-1][2])
        print(f"wd={wd}: held-out loss {np.mean(finals):.3f} +- {np.std(finals):.3f}  runs {[round(f,3) for f in finals]}")
    print("uniform", round(float(np.log(len(chars))), 3), "| batch seed varied:", has_seed)
