"""Study 2: capacity factor vs dropped tokens and accuracy."""
from common import *

SEEDS = range(6)
print("8 experts, 8 clusters, aux 0.01, mean±std over 6 seeds; capacity = ceil(cf * N * k / E)\n")
print("| k | capacity factor | drop rate | test MSE |")
print("|---|---|---|---|")
for k in (1, 2):
    for cf in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, None):
        rows = [trial(s, lambda sd: MoELayer(D, F, 8, k, cf, aux_coef=0.01, seed=sd)) for s in SEEDS]
        print(f"| {k} | {cf if cf else 'unlimited'} | {fmt(rows,'drop',3)} | {fmt(rows,'test')} |", flush=True)
