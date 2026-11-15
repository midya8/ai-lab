# boptlab

Bayesian optimisation from scratch in NumPy: a Gaussian-process surrogate with analytic marginal-likelihood gradients, EI / PI / UCB / Thompson-sampling acquisition, constant-liar batching, noisy objectives, and a benchmark study against random and quasi-random search. No SciPy, no BoTorch; every formula is tested against a brute-force or Monte-Carlo reference.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                 # 41 tests (~20 s)
python experiments/run_all.py regret kernels batch noise   # writes experiments/results_<section>.txt (~25 min on 2 cores; results.txt is those four files concatenated)
```

```python
import numpy as np
from boptlab import BayesOpt, hartmann6

bo = BayesOpt(hartmann6.bounds, acquisition="ei", rng=np.random.default_rng(0))
best_so_far = bo.run(hartmann6, n_total=60)        # history of the running minimum
x, y = bo.best()

# ask/tell interface for expensive external jobs, 4 points per round (constant liar)
X = bo.ask(q=4); bo.tell(X, [my_expensive_function(x) for x in X])
```

## What is in it

| module | contents |
|---|---|
| `gp.py` | ARD RBF and Matern-5/2 `Kernel` (with analytic d/dlog-lengthscale gradients, no 1/r singularity), `GP` (Cholesky, posterior mean/cov, joint samples, `condition` for fantasised points), `log_marginal_likelihood` with gradient, `fit_gp` (multi-start Adam in log-space, box-bounded, warm-startable) |
| `acquisition.py` | closed-form `expected_improvement` (with xi, exact zero-variance limit), `log_expected_improvement` (stable in the far tail; `acquisition='logei'`), `probability_of_improvement`, `confidence_bound`, Monte-Carlo EI and q-EI for checking |
| `optimizer.py` | `BayesOpt` (ask/tell/run; Halton initial design; EI/PI/UCB/TS; constant-liar batches; noise-aware incumbent), acquisition maximiser (random + Halton + near-incumbent candidates, then shrinking-radius local search), `halton`, `random_search` baselines |
| `costaware.py` | `CostAwareBayesOpt`: EI per unit cost (GP on log-cost) under a total budget; `best_at_budget` (unit-tested; no benchmark study yet) |
| `benchmarks.py` | Branin, Hartmann-3/6, Ackley, Levy, Rosenbrock with known minimisers, plus a `noisy` wrapper |

## What the tests check

- Kernels are PSD with `sf2` on the diagonal; the posterior equals the textbook formula with an explicit matrix inverse (mean, full covariance, variances); a noise-free GP interpolates and reverts to the prior far from data.
- `log_marginal_likelihood` equals the dense Gaussian log-density (`slogdet`) and its **gradient matches central finite differences** for both kernels, in every hyperparameter. The Matern gradient is finite at coincident points.
- Fitting raises the likelihood above the default start and recovers a lengthscale in the right range from GP-generated data. Posterior samples have the posterior mean and covariance (20,000 draws).
- EI equals Monte-Carlo EI (400k draws); EI is non-negative, monotone in the mean, increasing in sigma when the mean is worse than the incumbent, and finite in the deep tail; PI/UCB sign conventions; q-EI grows with batch size.
- Benchmark constants are validated by evaluating each function at its published minimisers (a typo in a Hartmann coefficient would show up here). The loop is deterministic for a seed, respects bounds, never reports below the optimum, and beats random search on Branin by 5x in median regret for EI, UCB and Thompson sampling. Batches return distinct points; noise-aware incumbents are smoothed.

## Results (8 seeds, final regret = f(best point) - optimum, median [IQR]; full tables in `experiments/results.txt`)

Matern-5/2 ARD GP, hyperparameters refit by marginal likelihood every iteration, UCB with beta = 2, 2d (min 5) Halton initial points that count toward the budget.

| function | budget | random | Halton | EI | UCB | Thompson |
|---|---|---|---|---|---|---|
| Branin (2-d) | 30 | 1.12 | 0.45 | **0.003** | 0.026 | 0.005 |
| Hartmann-3 | 40 | 0.45 | 0.32 | 0.004 | 0.008 | **0.000** |
| Hartmann-6 | 60 | 1.48 | 1.38 | **0.002** | 0.061 | 0.043 |
| Ackley-5 | 60 | 17.9 | 16.3 | 4.09 | **3.25** | 5.32 |
| Levy-4 | 50 | 5.69 | 3.67 | 2.27 | **1.01** | 3.32 |
| Rosenbrock-4 | 50 | 60.3 | 36.4 | **5.48** | 7.13 | 11.6 |

Reading it honestly:

- Every GP method beats both baselines by a large factor on every function; on Hartmann-6 EI finds the basin of the global optimum within 60 evaluations while random/Halton are ~1.4 away.
- **No acquisition function wins everywhere**: EI is best on Branin, Hartmann-6 and Rosenbrock, UCB on Ackley and Levy, Thompson on Hartmann-3. With 8 seeds the IQRs overlap for several of these pairs (e.g. EI vs UCB on Hartmann-6), so treat the ordering among the three GP methods as suggestive, not established. UCB's beta = 2 was not tuned; the others have no tunable setting.
- Ackley-5 and Levy-4 are not solved at this budget (regret 1 to 5): a stationary GP on a function with many ripples needs more data than 50-60 points in 4-5 dimensions.
- **Kernel choice** (EI, same seeds): Matern-5/2 beats RBF on Hartmann-6 (0.002 vs 0.087), Rosenbrock (5.5 vs 11.7) and Levy (2.3 vs 2.7); RBF wins on the very smooth Branin (0.000 vs 0.003). Small samples; direction only.
- **Batching** (Hartmann-6, 60 evals, constant liar): q=1 regret 0.002, q=4 0.010, q=8 0.147. Parallelising by 4 costs little; by 8 (only 6 sequential rounds) costs visibly.
- **Noise** (Branin + N(0,1) noise, 40 evals): EI's recommendation, measured on the *clean* function, has regret 0.29 when the incumbent is the best raw (lucky) observation and 0.18 with the noise-aware incumbent; recommending the GP posterior-mean argmin gives 0.17 / 0.16. Random search: 1.06. The naive-vs-aware gap is inside the IQRs at 8 seeds, so the safe conclusion is only that smoothing the recommendation helps and that EI is far ahead of random under noise.

## Caveats

- Everything is synthetic, low-dimensional, and uses the same seeds across methods; this is not a claim about hyperparameter tuning of real models or about dimensions above ~6.
- The Halton baseline initially used one fixed skip, so all seeds produced the same points; it now randomises the skip per seed (regression test included) and the table above is from the fixed code.
- Acquisition maximisation is sampling plus local search, not gradient-based; in higher dimensions this would be the first thing to replace.
- GP hyperparameters are point estimates (no MCMC / marginalisation), and there is no input warping or trust region.

## Constrained optimisation

`ConstrainedBayesOpt(bounds, n_constraints)` minimises f subject to c_k(x) <= 0: one GP per function, score = EI over the best feasible point x product of P(c_k <= 0); before any feasible point it maximises feasibility probability. Toy test: optimum on a disc boundary is found with the constraint respected (see `tests/test_constrained.py`; a 2-D toy, not a benchmark study).

## Max-value Entropy Search (`acquisition="mes"`)

`max_value_entropy_search(mu, sd, ystar)` implements MES (Wang & Jegelka 2017) for minimisation. The minimum-value samples `ystar` come from joint GP posterior draws on 400 random candidates (capped at the incumbent), not the Gumbel approximation of the paper. Gamma is clipped at -10 and the CDF uses erfc to avoid underflow. Tests check non-negativity, that an uncertain near-optimal point outscores a certain bad one, and that the loop gets Branin below 1.5 in 25 evaluations on one seed. It has not been benchmarked against EI/UCB/Thompson across seeds, so no ranking is claimed.

## Multi-seed acquisition comparison (12 fresh seeds)

`experiments/acq_compare.py` (output in `experiments/acq_compare_results.txt`) runs EI, log-EI, PI, UCB, Thompson and MES on seeds disjoint from `run_all.py`, 12 seeds each.

- **Branin (25 evals):** MES median regret 0.001 vs EI 0.022, and beats EI on 11/12 seeds; UCB (beta=2, untuned) and PI are worse.
- **Hartmann-3 (30 evals):** no clear winner; medians are all <= 0.015 and the IQRs overlap (Thompson 0.000, EI 0.001, MES 0.008).
- **log-EI is identical to EI** here (0/12 seeds differ): they share an argmax away from the far tail, so log-EI's benefit is numerical stability, not better results.
- Only two benchmarks and 12 seeds: the Branin MES gain looks real, the rest is not established.

**Harder problems** (`EXTRA=1 python experiments/acq_compare.py`, 8 seeds, output in `experiments/acq_compare_extra_results.txt`):

- **Hartmann-6 (50 evals):** EI median regret 0.003 vs UCB 0.018, Thompson 0.037, MES 0.124, PI 0.141; the IQRs of EI/UCB overlap. MES, the Branin winner, is among the worst here, so its Branin gain does not generalise.
- **Ackley-5 (50 evals):** all methods are far from the optimum; UCB is best (3.8) and beats EI on 7/8 seeds, EI 5.6. This fits the earlier finding that UCB does best on Ackley.
- 8 seeds only; ordering is suggestive.
