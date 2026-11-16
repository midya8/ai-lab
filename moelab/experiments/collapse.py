"""Study 1: does the load-balancing loss prevent routing collapse, and what does it cost?"""
from common import *

SEEDS = range(8)
print("8 experts, 8 clusters, capacity factor 1.25, mean±std over 8 task/init seeds\n")
for skew in (False, True):
    print(f"cluster frequencies: {'Zipf(1) skewed' if skew else 'uniform'}")
    print("| k | aux coef | test MSE | load CV | dead experts | drop rate | routing NMI |")
    print("|---|---|---|---|---|---|---|")
    for k in (1, 2):
        for aux in (0.0, 0.001, 0.01, 0.1, 1.0):
            rows = [trial(s, lambda sd: MoELayer(D, F, 8, k, 1.25, aux_coef=aux, seed=sd), skew=skew)
                    for s in SEEDS]
            print(f"| {k} | {aux} | {fmt(rows,'test')} | {fmt(rows,'cv',2)} | {fmt(rows,'dead',1)} | "
                  f"{fmt(rows,'drop',3)} | {fmt(rows,'nmi',2)} |", flush=True)
    print()
