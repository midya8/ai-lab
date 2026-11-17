"""Cost-aware EI (EIpu) vs cost-blind EI at equal *budget*, on Branin with a position-dependent cost.
cost(x) = 0.5 + 9.5 * u0^2, u0 = x0 normalised to [0,1] (so 0.5 on the left edge, 10 on the right; the three Branin
minimisers sit at u0 = 0.12, 0.50, 0.77 -> costs 0.64, 2.9, 6.1). Cost-model and budgets were fixed before running.
Usage: python experiments/cost_compare.py -> markdown table: regret at budget checkpoints, median [IQR] over seeds,
and the number of seeds where cost-aware is strictly better."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bayesopt import branin
from bayesopt.costaware import CostAwareBayesOpt, best_at_budget

SEEDS = range(200, 210)
BUDGETS = [15, 30, 60]


def cost_fn(x):
    u0 = (x[0] - branin.bounds[0, 0]) / (branin.bounds[0, 1] - branin.bounds[0, 0])
    return 0.5 + 9.5 * u0 ** 2


def med_iqr(a):
    a = np.asarray(a)
    return f"{np.median(a):.3f} [{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]"


if __name__ == "__main__":
    t0 = time.time()
    R = {True: {b: [] for b in BUDGETS}, False: {b: [] for b in BUDGETS}}
    N = {True: [], False: []}
    for s in SEEDS:
        for aware in (True, False):
            bo = CostAwareBayesOpt(branin.bounds, budget=max(BUDGETS), rng=np.random.default_rng(s), cost_aware=aware)
            tr = bo.run(branin, cost_fn)
            N[aware].append(len(bo.y))
            for b in BUDGETS:
                R[aware][b].append(best_at_budget(tr, b) - branin.optimum)
    print(f"Branin regret at equal cumulative cost, {len(SEEDS)} seeds, median [IQR] (inf = nothing affordable yet).\n")
    print("| budget | cost-aware (EIpu) | cost-blind EI | seeds aware strictly better | seeds tied |\n|---|---|---|---|---|")
    for b in BUDGETS:
        a, c = np.array(R[True][b]), np.array(R[False][b])
        print(f"| {b} | {med_iqr(a)} | {med_iqr(c)} | {int((a < c).sum())}/{len(a)} | {int((a == c).sum())} |")
    print(f"\nMedian evaluations used by budget {max(BUDGETS)}: aware {int(np.median(N[True]))}, blind {int(np.median(N[False]))}")
    print(f"({time.time()-t0:.0f}s)")
