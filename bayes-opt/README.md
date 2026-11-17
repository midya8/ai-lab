# bayes-opt

Bayesian optimisation in NumPy, no SciPy or BoTorch. A GP surrogate (ARD RBF or Matern-5/2, hyperparameters fit by marginal likelihood with analytic gradients), the usual acquisition functions (EI, log-EI, PI, UCB, Thompson, MES), constant-liar batches, noisy objectives, a cost-aware variant, a constrained variant and ParEGO for two objectives.

I wanted to see how much of the "BO beats random search" story holds up on the standard toy functions once you actually run it across seeds.

```bash
cd bayes-opt
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                         # ~40 s
python experiments/run_all.py regret kernels batch noise    # the big table, ~25 min
```

```python
import numpy as np
from bayesopt import BayesOpt, hartmann6

bo = BayesOpt(hartmann6.bounds, acquisition="ei", rng=np.random.default_rng(0))
bo.run(hartmann6, n_total=60)
x, y = bo.best()

X = bo.ask(q=4)                                    # ask/tell for expensive jobs
bo.tell(X, [my_expensive_function(x) for x in X])
```

## Results

Median final regret over 8 seeds (full tables in `experiments/results.txt`):

| function | budget | random | Halton | EI | UCB | Thompson |
|---|---|---|---|---|---|---|
| Branin | 30 | 1.12 | 0.45 | 0.003 | 0.026 | 0.005 |
| Hartmann-3 | 40 | 0.45 | 0.32 | 0.004 | 0.008 | 0.000 |
| Hartmann-6 | 60 | 1.48 | 1.38 | 0.002 | 0.061 | 0.043 |
| Ackley-5 | 60 | 17.9 | 16.3 | 4.09 | 3.25 | 5.32 |
| Levy-4 | 50 | 5.69 | 3.67 | 2.27 | 1.01 | 3.32 |
| Rosenbrock-4 | 50 | 60.3 | 36.4 | 5.48 | 7.13 | 11.6 |

Every GP method beats both baselines by a wide margin, but no acquisition function wins everywhere, and with 8 seeds a lot of the IQRs overlap. Ackley and Levy aren't really solved at these budgets.

Some smaller findings:

- Matern-5/2 beat RBF everywhere except the very smooth Branin.
- Batching on Hartmann-6: q=1 gets 0.002, q=4 gets 0.010, q=8 gets 0.147. Four in parallel is cheap, eight starts to hurt.
- MES won Branin clearly (0.001 vs EI 0.022, 11/12 seeds) and then did badly on Hartmann-6, so that win doesn't generalise.
- log-EI picked the same points as EI on every seed. It's a numerical fix, not a better strategy.
- Cost-aware EI (EI per unit cost) on Branin with a made-up cost surface: median regret 0.039 vs 2.06 at the same total spend.

## Limitations

Everything is synthetic and at most 6-D. The acquisition maximiser is random/Halton candidates plus local search, which is the first thing I'd replace in higher dimensions. GP hyperparameters are point estimates, no MCMC.
