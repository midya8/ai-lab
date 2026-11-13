# rlgrid

Tabular reinforcement learning from scratch (numpy only): a slippery gridworld with walls and a pit, Q-learning, SARSA, Dyna-Q, and value iteration as exact ground truth.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python experiments/compare.py
```

## Results (5 seeds, slip 0.1, 300 episodes; hyperparameters fixed in advance)

| agent | mean return, first 50 episodes | greedy success |
|---|---|---|
| Q-learning | +0.35 | 1.00 |
| SARSA | +0.33 | 1.00 |
| Dyna-Q (10 planning steps) | +0.65 | 1.00 |

Dyna-Q's model-based replay roughly doubles early return; all three reach the goal eventually. On this small map Q-learning and SARSA are indistinguishable; the classic cliff-style safety gap needs a map where the shortest path hugs a pit. Tests check that learned greedy return equals value iteration's optimum on the deterministic map.

## Cliff walking: the safety gap

`experiments/cliff.py` (map `CLIFF_MAP`, pits along the bottom edge, eps 0.1, 500 episodes, 10 seeds, parameters fixed before running):

| agent | online return (last 100 eps) | fell in pit | greedy path length |
|---|---|---|---|
| Q-learning | +0.529 | 17.1% | 13 |
| SARSA | +0.637 | 4.9% | 18 |

Q-learning learns the shorter edge-hugging route, but its epsilon-greedy exploration drops it into the pit more often while training; SARSA learns the longer, safer route and scores higher online. Greedy (no exploration) Q-learning's path is the shorter one.

Added afterwards (same settings, no tuning; added after seeing the two rows above):

| agent | online return | fell in pit | greedy path length |
|---|---|---|---|
| Expected SARSA | +0.799 | 1.9% | 15 |
| Double Q-learning | +0.811 | 0.4% | 17 |

Both beat SARSA online here. Double Q's low fall rate is plausibly from reduced maximisation bias, but I did not ablate that, so it is a hypothesis, not a finding.

Toy environment only; no claim about deep RL.

## REINFORCE

`reinforce(env, baseline=True)` is a tabular softmax policy-gradient agent (Monte-Carlo returns, optional learned V(s) baseline). `greedy_policy(theta)` extracts the argmax policy. Tests check it solves the deterministic gridworld; the baseline test is a loose sanity check (4 seeds), not a significance claim.

## Eligibility traces and n-step returns

`sarsa_lambda(env, lam=..., watkins=False, trace="replacing")` is SARSA(lambda) (set `watkins=True` for Watkins's Q(lambda), which cuts traces after exploratory actions); `n_step_sarsa(env, n=...)` is on-policy n-step SARSA. Tests pin the limits: lambda=0 reproduces SARSA / Q-learning and n=1 reproduces SARSA exactly (same RNG stream).

`experiments/traces.py` (default grid, slip 0.1, 8 seeds x 60 episodes, alpha=0.3 and eps=0.1 fixed beforehand; mean return over the 60 episodes):

| method | value |
|---|---|
| SARSA(lambda) lam 0 / 0.3 / 0.6 / 0.8 / 0.9 | +0.318 / +0.456 / +0.527 / +0.482 / +0.445 |
| Watkins Q(lambda) lam 0 / 0.3 / 0.6 / 0.8 / 0.9 | +0.296 / +0.438 / +0.440 / +0.427 / +0.403 |
| n-step SARSA n 1 / 2 / 4 / 8 | +0.326 / +0.485 / +0.535 / +0.422 |

Intermediate lambda / n learn fastest early; the extremes lose (lam 0 / n 1 propagate credit slowly, n 8 is noisier). Caveat: single environment, no confidence intervals, and n-step SARSA with n=4 left a greedy policy that loops against a wall on one of four test seeds (rarely-visited states keep optimistic zero values).

## Prioritized sweeping

`prioritized_sweeping(env, planning=10, theta=1e-4)` keeps a last-seen deterministic model and a priority queue of state-action pairs by |TD error|, backing up the largest first and pushing their predecessors. It returns `(q, returns, n_backups)`. `experiments/sweeping.py` (default grid, slip 0, 10 seeds, library-default hyperparameters, planning=10 for both planners) measures episodes until the greedy policy first matches value iteration's optimum:

| agent | episodes to optimal (mean) |
|---|---|
| Q-learning | 19.0 |
| Dyna-Q | 4.1 |
| prioritized sweeping | 3.0 |

Caveats: the gap between Dyna-Q and sweeping is one episode on a tiny deterministic map; I did not count Dyna-Q's backups, so this says nothing about compute efficiency (sweeping used about 1300 backups on average), and no significance test was run.
