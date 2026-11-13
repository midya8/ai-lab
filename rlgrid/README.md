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
