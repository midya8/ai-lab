"""Regenerates experiments/results.txt (concatenate the section outputs). Usage: python experiments/run_all.py <section> [<section> ...]
sections: regret, kernels, batch, noise.  Everything is seeded; regret = f(best point) - known optimum (clean f)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from boptlab import BayesOpt, BENCHMARKS, random_search, noisy, branin

OUT = os.path.dirname(__file__)
SEEDS = range(8)


def med_iqr(a):
    a = np.asarray(a)
    return f"{np.median(a):.3f} [{np.percentile(a, 25):.3f}, {np.percentile(a, 75):.3f}]"


def run_method(bench, method, n, seed, kernel="matern52", q=1):
    rng = np.random.default_rng(1000 + seed)
    if method in ("random", "halton"):
        _, y = random_search(bench, bench.bounds, n, rng, quasi=(method == "halton"))
        return np.minimum.accumulate(y) - bench.optimum
    bo = BayesOpt(bench.bounds, method, kernel=kernel, rng=rng)
    return bo.run(bench, n, q=q) - bench.optimum


def section_regret():
    plan = [("branin", 30), ("hartmann3", 40), ("hartmann6", 60), ("ackley5", 60), ("levy4", 50), ("rosenbrock4", 50)]
    lines = ["Final regret after the stated budget, median [IQR] over 8 seeds (lower is better; EI/UCB(beta=2)/TS use a Matern-5/2 ARD GP)", ""]
    lines.append("| function | budget | random | halton | EI | UCB | TS |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, n in plan:
        b = BENCHMARKS[name]
        cells = []
        for m in ["random", "halton", "ei", "ucb", "ts"]:
            cells.append(med_iqr([run_method(b, m, n, s)[-1] for s in SEEDS]))
            print(name, m, cells[-1], flush=True)
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    return lines


def section_kernels():
    lines = ["EI with RBF vs Matern-5/2, final regret median [IQR] over 8 seeds", "", "| function | budget | rbf | matern52 |", "|---|---|---|---|"]
    for name, n in [("branin", 30), ("hartmann6", 60), ("rosenbrock4", 50), ("levy4", 50)]:
        b = BENCHMARKS[name]
        cells = [med_iqr([run_method(b, "ei", n, s, kernel=k)[-1] for s in SEEDS]) for k in ("rbf", "matern52")]
        print(name, cells, flush=True)
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    return lines


def section_batch():
    b, n = BENCHMARKS["hartmann6"], 60
    lines = ["Hartmann6, 60 evaluations total, EI with constant-liar batches of size q; regret median [IQR] over 8 seeds",
             "(fewer sequential rounds for larger q, the price is information)", "", "| q | rounds | regret |", "|---|---|---|"]
    for q in (1, 4, 8):
        r = [run_method(b, "ei", n, s, q=q)[-1] for s in SEEDS]
        rounds = (n - b.dim * 2 + q - 1) // q
        print(q, med_iqr(r), flush=True)
        lines.append(f"| {q} | {rounds} | {med_iqr(r)} |")
    return lines


def section_noise():
    n, sd = 40, 1.0
    lines = [f"Branin with N(0, {sd}^2) observation noise, {n} evaluations, 8 seeds. 'true regret' = clean f at the point the method would",
             "recommend minus optimum. Raw: recommend the best observed point. Smoothed: recommend the argmin of the GP posterior mean over visited points.", "",
             "| optimiser | recommendation | true regret median [IQR] |", "|---|---|---|"]
    rows = {}
    for label, noisy_flag in (("EI, noise-naive incumbent", False), ("EI, noise-aware incumbent", True)):
        raw, smooth = [], []
        for s in SEEDS:
            nb = noisy(branin, sd, np.random.default_rng(500 + s))
            bo = BayesOpt(nb.bounds, "ei", noisy=noisy_flag, rng=np.random.default_rng(2000 + s))
            bo.run(nb, n)
            raw.append(branin(bo.X[int(np.argmin(bo.y))]) - branin.optimum)
            gp = bo._fit()
            smooth.append(branin(bo.X[int(np.argmin(gp.predict(bo.U)[0]))]) - branin.optimum)
        rows[label] = (raw, smooth)
        print(label, med_iqr(raw), med_iqr(smooth), flush=True)
        lines.append(f"| {label} | raw best observation | {med_iqr(raw)} |")
        lines.append(f"| {label} | smoothed (posterior mean) | {med_iqr(smooth)} |")
    rr = []
    for s in SEEDS:
        nb = noisy(branin, sd, np.random.default_rng(500 + s))
        X, y = random_search(nb, nb.bounds, n, np.random.default_rng(3000 + s))
        rr.append(branin(X[int(np.argmin(y))]) - branin.optimum)
    lines.append(f"| random search | raw best observation | {med_iqr(rr)} |")
    return lines


if __name__ == "__main__":
    for sec in sys.argv[1:]:
        t = time.time()
        lines = globals()["section_" + sec]()
        lines.append(f"\n(runtime {time.time() - t:.0f} s)")
        open(os.path.join(OUT, f"results_{sec}.txt"), "w").write("\n".join(lines) + "\n")
