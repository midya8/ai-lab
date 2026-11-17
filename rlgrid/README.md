# rlgrid

Tabular reinforcement learning on a slippery gridworld, NumPy only. Value iteration gives the exact answer, so every learner can be scored by its true regret instead of noisy rollouts. It started with Q-learning, SARSA and Dyna-Q and kept growing: eligibility traces, n-step SARSA, prioritized sweeping, REINFORCE, linear actor-critic, and a small NumPy DQN with replay, target nets, Double, dueling and prioritized replay.

```bash
cd rlgrid
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python experiments/compare.py
PYTHONPATH=src python experiments/cliff.py
PYTHONPATH=src python experiments/regret_curves.py
```

## Results worth keeping

Cliff walking (10 seeds, epsilon 0.1):

| agent | online return | fell in the pit | greedy path |
|---|---|---|---|
| Q-learning | +0.529 | 17.1% | 13 |
| SARSA | +0.637 | 4.9% | 18 |
| Expected SARSA | +0.799 | 1.9% | 15 |
| Double Q-learning | +0.811 | 0.4% | 17 |

The textbook result shows up. Q-learning learns the short route along the edge and keeps falling off it while exploring. SARSA takes the longer, safer route.

Other things:

- Dyna-Q's planning roughly doubles early return, and gives ~10x lower regret than the model-free learners at small budgets.
- Middle values of lambda and n learn fastest early on. The extremes are slower or noisier.
- Linear actor-critic on raw (row, col) coordinates fails completely, since a walled maze isn't linear in position. RBF features work well.
- For DQN, removing either replay or the target network alone did nothing measurable here. Removing both dropped success to 60%. Double DQN and prioritized replay made no difference on a grid this small.

All toy-sized, with 5-10 seeds and no confidence intervals. Some DQN hyperparameters were picked on overlapping seeds, so those numbers are optimistic.
