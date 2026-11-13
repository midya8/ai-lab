# mctslab

Monte-Carlo tree search (UCT, random rollouts) written from scratch, validated against an exact minimax solver on small games.

- `ConnectN(rows, cols, k, gravity)`: tic-tac-toe (default) up to Connect-Four-style boards.
- `uct_search` / `MCTS`: returns the most-visited move plus visit counts.
- `minimax_value` / `best_moves`: memoised exact ground truth (practical for tiny boards only).
- `arena.match`: alternating-colour matches vs random or perfect players.

```python
from mctslab import TicTacToe, MCTS
g = TicTacToe(); print(MCTS(g, 1000)(g.initial()))
```

Test: `PYTHONPATH=src python -m pytest -q` (7 tests, ~12 s). Tests check that MCTS takes wins, blocks threats, never loses to random play or draws perfect play in the small samples used, and agrees with minimax on >=80% of sampled positions at 1000 iterations. Samples are small; this is a correctness check, not a strength benchmark. No learned policy/value network yet (see ROADMAP).
