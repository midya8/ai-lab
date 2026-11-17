"""Study 2b: where does recovery finally break? Same as phase.py but with a denser true L0
(12 and 24 active features out of 64-256, d=32) and BatchTopK only; latents = 2 x features; 3 seeds."""
from common import *

D, SEEDS = 32, range(3)
print("d=32; BatchTopK with k = true L0; mean±std over 3 seeds")
print(f"{'feat/d':>7}{'trueL0':>7}{'matched':>14}{'mean_cos':>14}{'FVU':>14}{'L0':>12}", flush=True)
for ratio in (2, 4, 8):
    n = int(ratio * D)
    for l0_true in (12, 24):
        rows = []
        for s in SEEDS:
            w = make_world(n, D, l0=l0_true, seed=s)
            sae = SAE(D, 2 * n, mode="batchtopk", k=l0_true, seed=s)
            fit(sae, w, steps=3000, seed=s)
            rows.append(evaluate(sae, w, s, n=4096))
        print(f"{ratio:>7}{l0_true:>7}{fmt(rows, 'matched'):>14}{fmt(rows, 'mean_cos'):>14}{fmt(rows, 'FVU'):>14}{fmt(rows, 'L0', 1):>12}", flush=True)
