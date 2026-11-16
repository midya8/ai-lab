"""Study 3: dead latents and resampling. 64 true features, d=32, true L0=3, BatchTopK k=3 and
TopK k=3; more latents than features; resample every 300 steps until 60% of training; 4 seeds."""
from common import *

SEEDS = range(4)
print("d=32, 64 true features, k=3, 3000 steps; mean±std over 4 seeds")
print(f"{'mode':<10}{'latents':>8}{'resample':>9}{'dead':>14}{'matched':>14}{'purity(alive)':>16}{'FVU':>14}", flush=True)
for mode in ("topk", "batchtopk"):
    for m in (64, 128, 256, 512):
        for rs in (0, 300):
            rows = []
            for s in SEEDS:
                w = make_world(64, 32, l0=3, seed=s)
                sae = SAE(32, m, mode=mode, k=3, seed=s)
                fit(sae, w, steps=3000, seed=s, resample_every=rs)
                rows.append(evaluate(sae, w, s))
            print(f"{mode:<10}{m:>8}{('yes' if rs else 'no'):>9}{fmt(rows, 'dead', 1):>14}{fmt(rows, 'matched'):>14}{fmt(rows, 'purity'):>16}{fmt(rows, 'FVU'):>14}", flush=True)
