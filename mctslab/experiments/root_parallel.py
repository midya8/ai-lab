"""Same total budget: 1 tree x N vs 4 trees x N/4, versus a perfect player (tic-tac-toe)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mctslab import TicTacToe, MCTS
from mctslab.root_parallel import RootParallelMCTS
from mctslab.arena import match, perfect_player, random_player

g = TicTacToe()
for budget in (20, 40, 80):
    for name, mk in (("single", lambda b, s: MCTS(g, b, seed=s)),
                     ("root4", lambda b, s: RootParallelMCTS(g, b, 4, seed=s))):
        tot = {"wins": 0, "draws": 0, "losses": 0}
        for s in range(5):
            r = match(g, mk(budget, s), perfect_player(g, s), games=10)
            for k in tot: tot[k] += r[k]
        print(f"budget={budget:3d} {name:6s} vs perfect (50 games): {tot}")
