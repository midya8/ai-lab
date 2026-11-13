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

Toy environment only; no claim about deep RL.
