"""Study 4: auxiliary-loss-free (selection-bias) balancing vs the Switch balance loss."""
from common import *

SEEDS = range(8)
CONFIGS = [("none", dict()), ("aux 0.01", dict(aux_coef=0.01)), ("aux 0.1", dict(aux_coef=0.1)),
           ("bias 0.001", dict(bias_rate=0.001)), ("bias 0.003", dict(bias_rate=0.003)),
           ("bias 0.01", dict(bias_rate=0.01)), ("bias 0.03", dict(bias_rate=0.03))]
print("8 experts, 8 clusters, capacity factor 1.25, mean±std over 8 task/init seeds\n")
for skew in (False, True):
    print(f"cluster frequencies: {'Zipf(1) skewed' if skew else 'uniform'}")
    print("| k | balancing | test MSE | load CV | dead experts | drop rate | routing NMI |")
    print("|---|---|---|---|---|---|---|")
    for k in (1, 2):
        for name, kw in CONFIGS:
            rows = [trial(s, lambda sd: MoELayer(D, F, 8, k, 1.25, seed=sd, **kw), skew=skew)
                    for s in SEEDS]
            print(f"| {k} | {name} | {fmt(rows,'test')} | {fmt(rows,'cv',2)} | {fmt(rows,'dead',1)} | "
                  f"{fmt(rows,'drop',3)} | {fmt(rows,'nmi',2)} |", flush=True)
    print()
