# mcts

Monte-Carlo tree search (UCT with random rollouts) for tic-tac-toe and Connect-N, checked against an exact minimax solver on the small boards. Later additions: PUCT with a pluggable prior/value, root-parallel search, a transposition table, and an AlphaZero-lite policy/value net trained from self-play.

```python
from treesearch import TicTacToe, MCTS
g = TicTacToe()
print(MCTS(g, 1000)(g.initial()))
```

```bash
cd mcts
PYTHONPATH=src python -m pytest -q                    # ~20 s
python experiments/azlite_selfplay.py 200             # self-play training, ~30 s
python experiments/strength_ladder.py                 # Connect-Four, budget vs budget
```

## Results

How often UCT picks a minimax-optimal move on 300 tic-tac-toe positions: 0.87 at 10 iterations, 0.97 at 50, 0.987 at 200, 0.997 at 1000. Its value estimates converge too (mean error 0.17 at 25 iterations, 0.047 at 1600).

AlphaZero-lite on tic-tac-toe, 15 search iterations per move, 40 games each:

| matchup | W / D / L |
|---|---|
| vs random | 36 / 4 / 0 |
| vs plain UCT | 29 / 10 / 1 |
| vs perfect minimax | 0 / 38 / 2 |

For comparison, plain UCT at the same budget loses 28 of 40 to the perfect player. With only 30 training rounds instead of 200 the net actually lost to UCT, so it needs enough self-play.

Things that didn't help: the transposition table made no difference on tic-tac-toe, and splitting a small budget across 4 root-parallel trees was slightly worse (it only buys wall-clock time). On Connect-Four, PUCT with a hand-written open-line heuristic beat plain UCT 12-0 at 60 iterations. That's mostly "a cheap evaluator beats random rollouts at tiny budgets", which isn't surprising.

Sample sizes are small throughout (10-50 games per cell, mostly one seed), so read these as trends. Gravity boards aren't supported by AlphaZero-lite yet.
