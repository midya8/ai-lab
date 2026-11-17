"""Multi-seed acquisition comparison including MES and log-EI (fresh seeds 100..111, disjoint from run_all.py).
Usage: python experiments/acq_compare.py  -> prints a markdown table (final regret median [IQR], plus wins vs EI)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bayesopt import BayesOpt, branin, hartmann3, hartmann6
from bayesopt.benchmarks import ackley

SEEDS = range(100, 112)
ACQS = ["ei", "logei", "pi", "ucb", "ts", "mes"]
BENCH = [(branin, 25), (hartmann3, 30)]
if os.environ.get('EXTRA') == '1':   # harder problems, run separately: EXTRA=1 python experiments/acq_compare.py
    BENCH = [(hartmann6, 50), (ackley(5), 50)]
    SEEDS = range(100, 108)


def med_iqr(a):
    a = np.asarray(a)
    return f"{np.median(a):.3f} [{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]"


if __name__ == "__main__":
    t0 = time.time()
    print(f"Final regret (f(best) - optimum), {len(SEEDS)} fresh seeds, median [IQR]. 'beats EI' = seeds where regret is strictly lower than EI on the same seed.\n")
    print("| benchmark (evals) | " + " | ".join(ACQS) + " |\n|---|" + "---|" * len(ACQS))
    for b, n in BENCH:
        R = {a: [] for a in ACQS}
        for s in SEEDS:
            for a in ACQS:
                bo = BayesOpt(b.bounds, a, rng=np.random.default_rng(1000 + s))
                R[a].append(bo.run(b, n)[-1] - b.optimum)
        print(f"| {b.name} ({n}) | " + " | ".join(med_iqr(R[a]) for a in ACQS) + " |", flush=True)
        print(f"| &nbsp; beats EI | " + " | ".join("-" if a == "ei" else f"{sum(x < e for x, e in zip(R[a], R['ei']))}/{len(SEEDS)}" for a in ACQS) + " |", flush=True)
    print(f"\n(runtime {time.time()-t0:.0f} s)")
