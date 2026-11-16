"""Study 5: shared experts and fine-grained experts at a fixed active-parameter budget."""
from common import *

SEEDS = range(8)
# every variant activates 64 expert hidden units per token and holds 256 in total
VARIANTS = [
    ("top-2 of 8 (f=32)", 8, 32, 2, 0),
    ("1 shared + top-1 of 8 (f=32)", 8, 32, 1, 1),
    ("top-4 of 16 (f=16)", 16, 16, 4, 0),
    ("1 shared + top-3 of 16 (f=16)", 16, 16, 3, 1),
    ("top-8 of 32 (f=8)", 32, 8, 8, 0),
    ("1 shared + top-7 of 32 (f=8)", 32, 8, 7, 1),
]
print("aux 0.01, capacity factor 1.25, mean±std over 8 seeds; all variants: 64 active expert hidden units\n")
for nc in (8, 16, 32):
    print(f"clusters: {nc}")
    print("| variant | active params/token | test MSE | load CV | dead experts | drop rate | routing NMI |")
    print("|---|---|---|---|---|---|---|")
    for name, E, f, k, ns in VARIANTS:
        rows = [trial(s, lambda sd: MoELayer(D, f, E, k, 1.25, aux_coef=0.01, n_shared=ns, seed=sd),
                      n_clusters=nc) for s in SEEDS]
        print(f"| {name} | {rows[0]['active']} | {fmt(rows,'test')} | {fmt(rows,'cv',2)} | "
              f"{fmt(rows,'dead',1)} | {fmt(rows,'drop',3)} | {fmt(rows,'nmi',2)} |", flush=True)
    print()
