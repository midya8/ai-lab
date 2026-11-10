"""Speculative sampling with REAL (tiny, trained) transformers from microgrid.

Target: MiniGPT d=48, 2 layers. Draft: MiniGPT d=24, 1 layer. Both trained on
the same stdlib-source char corpus (held-out files for the prompt text).
Reports empirical acceptance, tokens per target call, and the cost ratio c
measured as wall-clock of one forward pass (draft/target), then the modelled
cost per token = (target_calls + c*proposed)/tokens vs autoregressive = 1.0.
Numpy autodiff models, single-position forward, no KV cache: c is whatever that gives."""
import os, sys, time, glob, sysconfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
from specdec.neural import MiniGPTLM, n_params
from specdec import speculative_generate
from microgrid import MiniGPT
from microgrid.lm import make_data_split, train

CTX, STEPS, N = 32, int(sys.argv[1]) if len(sys.argv) > 1 else 300, 120


def load(files, limit):
    out = []
    for f in files:
        t = open(f, encoding="utf-8", errors="ignore").read()
        if t.isascii():
            out.append(t)
        if sum(map(len, out)) >= limit:
            break
    return "".join(out)[:limit]


files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
chars, tr, va = make_data_split(load(files[0::2], 50000), load(files[1::2], 8000))
V = len(chars)
tm = MiniGPT(V, d=48, heads=4, layers=2, ctx=CTX, seed=0)
dm = MiniGPT(V, d=24, heads=4, layers=1, ctx=CTX, seed=1)
for m in (tm, dm):
    t0 = time.time()
    h = train(m, tr, va, CTX, steps=STEPS, bs=16, lr=3e-3, log_every=STEPS, clip=1.0, warmup=20, schedule=True, seed=0)
    print(f"trained {n_params(m)} params in {time.time()-t0:.0f}s, val nll {h[-1][2]:.3f}")
target, draft = MiniGPTLM(tm, CTX), MiniGPTLM(dm, CTX)
ctx = [int(x) for x in va[:CTX]]
def timeit(lm):
    t0 = time.time()
    for _ in range(60):
        lm.probs(ctx)
    return (time.time() - t0) / 60
c = timeit(draft) / timeit(target)
print(f"param ratio {n_params(dm)/n_params(tm):.2f}, measured forward-time ratio c = {c:.2f}")
print("   k  alpha_emp  tok/target_call  cost/token(c measured)  measured wall-clock vs AR")
prompt = [int(x) for x in va[:8]]
t0 = time.time()
for s in range(2):
    from specdec import autoregressive_generate
    autoregressive_generate(target, prompt, N, np.random.default_rng(s))
ar_t = (time.time() - t0) / 2
for k in (1, 2, 4, 6, 8):
    al = tp = cost = wt = 0.0
    S = 3
    for s in range(S):
        st = {}
        t0 = time.time()
        speculative_generate(target, draft, prompt, N, k, np.random.default_rng(s), st)
        wt += time.time() - t0
        al += st["accepted"] / st["proposed"]
        tp += N / st["target_calls"]
        cost += (st["target_calls"] + c * st["proposed"]) / N
    print(f"  {k:2d}  {al/S:8.3f}  {tp/S:14.2f}  {cost/S:18.3f}  {wt/S/ar_t:18.2f}x time")
print("note: this adapter calls the target once per verified position (no batched pass), so measured wall-clock >1 is expected;")
print("cost/token uses the modelled 'one batched target pass per round' accounting.")
