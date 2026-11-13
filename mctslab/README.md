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

- `tt_uct_search` / `TTMCTS`: UCT with a transposition table (statistics shared across move orders).

**Transposition result (null/negative, small sample).** On 59 sampled tic-tac-toe positions, fraction where the chosen move is minimax-optimal, plain UCT vs transposition UCT (same seeds, c=1.4, not tuned): 100 iters 49 vs 46, 300 iters 57 vs 44, 1000 iters 57 vs 54. Sharing did not help here; tic-tac-toe is too small for it to pay off and I did not investigate larger boards, so treat it as an unproven variant.

Test: `PYTHONPATH=src python -m pytest -q` (12 tests, ~13 s). Tests check that MCTS takes wins, blocks threats, never loses to random play or draws perfect play in the small samples used, and agrees with minimax on >=80% of sampled positions at 1000 iterations. Samples are small; this is a correctness check, not a strength benchmark. No learned policy/value network yet (see ROADMAP).

**Strength ladder (Connect-Four 6x7, 10 games per cell, alternating colours, draws = 0.5; `experiments/strength_ladder.py`).**

| iters | vs random | vs previous budget |
|---|---|---|
| 10 | 1.00 | - |
| 40 | 0.90 | 1.00 |
| 160 | 1.00 | 0.90 |
| 640 | 1.00 | 0.95 |

More search clearly helps, but 10 games per cell is a tiny sample (one game moves a cell by 0.10), so treat these as trends; even 10 iterations beats random play, and the 0.90 at 40 iterations vs random is plausibly noise.

**PUCT with pluggable prior/value (`puct.py`, `experiments/puct_vs_uct.py`).** AlphaZero-style selection (Q + c*P*sqrt(N)/(1+n)) with callables for the policy prior and value, so a learned net can be slotted in later; nothing is learned yet. Connect-Four 6x7, 60 iterations each, 12 games with alternating colours vs plain UCT: PUCT with a random-rollout value scored 0.46 (5W 1D 6L, i.e. no difference); PUCT with a hand-written open-line heuristic value scored 1.00 (12-0). Caveats: equal *iterations*, not equal wall-clock (the heuristic is much cheaper than a rollout, which favours it further); play is nearly deterministic given seeds, so 12 games are far fewer independent samples than they look; c_puct=1.5 and the heuristic scale were fixed in advance, not tuned. The result says a decent static evaluator beats random rollouts at low budgets, which is expected, not a claim about strength at large budgets. 19 tests total.

## Root-parallel UCT

`root_parallel_search` runs several independent trees (own seeds, split budget) and sums root visits. Results are identical sequentially or with `processes=True`. Experiment (`experiments/root_parallel.py`, tic-tac-toe vs perfect player, 50 games per cell, equal total budget): single tree lost 33/24/15 games at budgets 20/40/80; root-4 was slightly worse at every budget (losses 43/27/20). Splitting a small budget hurts; the benefit is wall-clock only.
