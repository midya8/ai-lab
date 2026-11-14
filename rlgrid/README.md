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

## Actor-critic with linear function approximation

`actor_critic(env, features="onehot"|"coords"|"rbf")` is a one-step online actor-critic: softmax policy `softmax(phi(s) @ theta)`, critic `V = phi(s) @ w` trained by TD(0), with the TD error as the advantage. `make_features` builds the feature maps; a custom `(n_states, d)` matrix also works. `experiments/actor_critic.py` (default grid, slip 0, 10 seeds, 300 episodes, step sizes 0.1/0.1 fixed a priori and not tuned per feature set):

| agent | final greedy success | mean training return |
|---|---|---|
| actor-critic one-hot | 1.0 | +0.023 |
| actor-critic coords (row, col) | 0.0 | -1.791 |
| actor-critic RBF sigma=0.7 | 1.0 | +0.432 |
| actor-critic RBF sigma=1.0 | 1.0 | +0.426 |
| REINFORCE + baseline (tabular) | 1.0 | +0.550 |

Findings: linear coordinates fail completely (a walled maze is not linearly separable in position), RBF features learn much faster than one-hot by sharing credit between neighbours, but tabular REINFORCE with a baseline still had the best mean return here. Caveats: tiny deterministic map, no confidence intervals, sigma values were two guesses, and one-step actor-critic with one-hot features was slowed by its small step size.

## DQN in numpy (`rlgrid.dqn`)

`dqn(env, ...)` trains a one-hidden-layer ReLU Q-network with uniform replay and an optional target network; `dqn_policy(net, phi)` gives the greedy policy (4 tests incl. a finite-difference gradient check). `python experiments/dqn_ablation.py` (default grid, 5 seeds, 300 episodes):

| variant | greedy success | last-50 train return |
|---|---|---|
| replay + target | 1.00 | +0.850 |
| no target | 1.00 | +0.851 |
| no replay | 1.00 | +0.837 |
| neither | 0.60 | -0.289 |

Findings, honestly: removing *either* stabiliser alone does nothing measurable on this tiny deterministic grid; only removing both hurts. Hyperparameters (lr 0.2, eps decay 0.99, hidden 64, 200-step cap) were chosen after a small search on seeds 0-2, which overlap the evaluation seeds, so treat the absolute numbers as optimistic. The first attempt (lr 0.05, decay 0.97, 100-step cap) never reached the goal: exploration collapsed before the sparse reward was found. Coordinate features (`features="coords"`) did not work with the small MLP here (0/5 success at default settings); one-hot is the default, which makes the net tabular-equivalent, so this shows the training loop is correct rather than any generalisation.

Pass `double=True` to `dqn()` for Double DQN: the online net chooses the next action and the target net scores it (`double_q_target`), which curbs max-operator overestimation. Tested for the target identity and for reaching the goal on a 3x3 grid; no benchmark claim is made that it beats vanilla here.

`q_bias(env, net, phi, qstar)` measures a network's mean signed error, RMSE and greedy agreement against exact Q* from value iteration. `python experiments/q_bias.py` (4x4 grid with walls, slip 0.2, 5 seeds, 300 episodes):

| variant | mean signed err | RMSE | greedy agreement |
|---|---|---|---|
| vanilla | -0.004 | 0.008 | 0.94 |
| double | -0.003 | 0.007 | 0.88 |

Null result: with reward clipping-free small rewards, step cost and a tiny state space, neither variant overestimates measurably, so Double DQN has no bias to fix here; the agreement gap is within what 5 seeds can resolve.

Pass `dueling=True` for a dueling network (`DuelQNet`: shared trunk, value and mean-centred advantage heads, Q = V + A - mean(A)); its hand-written backward pass is verified against finite differences. Reaches the goal on the 3x3 grid; no benchmark claim of improvement.

Pass `prioritized=True` for proportional prioritized replay (`per_alpha=0.6`, `per_beta=0.4`, new transitions get max priority, importance-sampling weights normalised by their max; `QNet.step` now takes optional `weights` and returns the raw TD error). `python experiments/per_ablation.py` (5 seeds, 300 episodes): uniform and prioritized both reach 1.00 greedy success and +0.850 last-50 return. Honest null result: the tiny deterministic grid is too easy for prioritisation to matter; alpha/beta are the paper defaults, untuned. Tests cover the weighting maths and determinism.

## Exact policy evaluation and regret

`policy_evaluation(env, policy)` computes V^pi exactly from the known model, and `policy_regret(env, policy)` returns per-state regret V* - V^pi and its mean. Unlike `evaluate()`, there is no sampling noise, so two learned policies can be compared without rollouts. Tests check zero regret for the optimal policy and positive regret for an all-up policy on a slippery map.

## Regret curves

`regret_curve(env, algo, checkpoints, seeds)` trains each learner for several episode budgets (independent runs) and reports the exact mean policy regret of the greedy policy, mean±std over seeds. `python experiments/regret_curves.py` (slip 0.1, 5 seeds, default hyperparameters, untuned):

| algo | 10 | 25 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| q_learning | 0.526±0.047 | 0.209±0.023 | 0.144±0.027 | 0.145±0.012 | 0.121±0.020 | 0.131±0.031 |
| sarsa | 0.511±0.042 | 0.172±0.028 | 0.196±0.068 | 0.245±0.153 | 0.124±0.039 | 0.091±0.043 |
| expected_sarsa | 0.497±0.039 | 0.199±0.020 | 0.141±0.020 | 0.143±0.044 | 0.141±0.014 | 0.138±0.018 |
| dyna_q(10) | 0.055±0.048 | 0.033±0.035 | 0.026±0.015 | 0.031±0.020 | 0.036±0.043 | 0.056±0.032 |

Findings: Dyna-Q's planning gives ~10x lower regret at small budgets. The model-free learners plateau near 0.13 because constant epsilon=0.1 and alpha=0.5 keep the greedy policy noisy; they do not converge to zero regret within 400 episodes. Dyna-Q's regret also drifts up slightly at 400, within seed noise.
