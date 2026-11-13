"""PUCT (rollout value / heuristic value) vs UCT at equal iteration budgets on Connect-Four 6x7.
c_puct and the heuristic scale were fixed a priori (1.5, tanh(score/4)); not tuned."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mctslab import ConnectN, MCTS, PUCT, line_heuristic_value
from mctslab.arena import match

g = ConnectN(6, 7, 4, gravity=True)
games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 60
for name, mk in [("PUCT rollout", lambda: PUCT(g, iters, seed=1)),
                 ("PUCT heuristic", lambda: PUCT(g, iters, value_fn=lambda gm, s: line_heuristic_value(gm, s), seed=1))]:
    r = match(g, mk(), MCTS(g, iters, seed=2), games=games)
    print(f"{name} vs UCT @{iters} iters, {games} games: {r} score={(r['wins']+0.5*r['draws'])/games:.2f}")
