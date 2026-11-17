"""Study 3: MoE routing schemes vs dense FFNs as the number of task clusters grows."""
from common import *

SEEDS = range(6)
print("8 experts (f=32 each), mean±std test MSE over 6 seeds; 400 full-batch Adam steps, N=1024\n")
cols = [
    ("top-1 (aux .02)", lambda sd: MoELayer(D, F, 8, 1, 1.25, aux_coef=0.02, seed=sd)),
    ("top-2 (aux .01)", lambda sd: MoELayer(D, F, 8, 2, 1.25, aux_coef=0.01, seed=sd)),
    ("expert-choice k=2", lambda sd: MoELayer(D, F, 8, 2, 1.0, routing="expert_choice", seed=sd)),
    ("dense, 2 experts wide", lambda sd: DenseFFN(D, 2 * F, seed=sd)),
    ("dense, 8 experts wide", lambda sd: DenseFFN(D, 8 * F, seed=sd)),
]
print("| clusters | " + " | ".join(c[0] for c in cols) + " |")
print("|---|" + "---|" * len(cols))
for K in (4, 8, 16, 32):
    cells = []
    for name, fn in cols:
        rows = [trial(s, fn, n_clusters=K) for s in SEEDS]
        cells.append(fmt(rows, "test"))
    print(f"| {K} | " + " | ".join(cells) + " |", flush=True)
print("\nparameters: top-k MoE", 17572, "| dense-2x", 4612, "| dense-8x", 17092, "(total); active per token: top-1 ~", 32*8+ (32*32*2+32+32), ", top-2 ~", 32*8 + 2*(32*32*2+32+32))
