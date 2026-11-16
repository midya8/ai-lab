"""Study 2: how much superposition can a SAE undo? Sweep features/d and true L0 (d=32).
TopK and BatchTopK use k = true L0; latents = 2 x features; 3 seeds."""
from common import *

D, SEEDS = 32, range(3)
print("d=32; matched = fraction of true features recovered one-to-one at cos>=0.9; mean±std over 3 seeds")
print(f"{'feat/d':>7}{'trueL0':>7}{'coher':>7}   {'topk matched':>14}{'topk FVU':>12}   {'batchtopk matched':>18}{'bt FVU':>12}", flush=True)
for ratio in (0.5, 1, 2, 4, 8):
    n = int(ratio * D)
    for l0_true in (1, 3, 6):
        out = {}
        coh = []
        for mode in ("topk", "batchtopk"):
            rows = []
            for s in SEEDS:
                w = make_world(n, D, l0=l0_true, seed=s)
                coh.append(w.coherence())
                sae = SAE(D, 2 * n, mode=mode, k=l0_true, seed=s)
                fit(sae, w, steps=3000, seed=s)
                rows.append(evaluate(sae, w, s, n=4096))
            out[mode] = rows
        print(f"{ratio:>7}{l0_true:>7}{np.mean(coh):>7.2f}   {fmt(out['topk'], 'matched'):>14}{fmt(out['topk'], 'FVU'):>12}   {fmt(out['batchtopk'], 'matched'):>18}{fmt(out['batchtopk'], 'FVU'):>12}", flush=True)
