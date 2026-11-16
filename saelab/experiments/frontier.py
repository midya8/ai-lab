"""Study 1: sparsity/fidelity frontier. L1-ReLU vs TopK vs BatchTopK on 64 true features in d=32
(2x superposition, true L0 = 3), 128 latents, 5 seeds (seed changes the world and the init)."""
from common import *

SEEDS = range(5)
CONFIGS = ([("relu", {"l1": v}) for v in (0.03, 0.1, 0.2, 0.4, 0.8)]
           + [("topk", {"k": k}) for k in (1, 2, 3, 4, 6, 8)]
           + [("batchtopk", {"k": k}) for k in (1, 2, 3, 4, 6, 8)])
print("d=32, 64 true features (true L0=3), 128 latents, 3000 steps x batch 512, 5 seeds; mean±std")
print(f"{'config':<18}{'L0':>12}{'FVU':>14}{'matched':>14}{'purity':>14}{'dead':>14}{'ratio':>14}{'recall':>14}")
for mode, kw in CONFIGS:
    rows = []
    for s in SEEDS:
        w = make_world(64, 32, l0=3, seed=s)
        sae = SAE(32, 128, mode=mode, seed=s, **kw)
        fit(sae, w, steps=3000, seed=s)
        rows.append(evaluate(sae, w, s))
    name = f"{mode} " + ",".join(f"{k}={v}" for k, v in kw.items())
    print(f"{name:<18}" + "".join(f"{fmt(rows, k, 2 if k in ('L0', 'dead') else 3):>14}" for k in ("L0", "FVU", "matched", "purity", "dead", "ratio", "recall")), flush=True)
