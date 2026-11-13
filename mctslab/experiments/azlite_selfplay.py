"""AlphaZero-lite on tic-tac-toe: train from PUCT self-play, then evaluate vs random, vs UCT and vs a
perfect minimax player at equal iteration budgets. Hyperparameters fixed a priori (see below)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mctslab import TicTacToe, PUCT, MCTS
from mctslab.azlite import PVNet, train
from mctslab.arena import match, random_player, perfect_player

g = TicTacToe()
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
iters = 15
t = time.time()
net = train(g, PVNet(g, 64, seed=0), rounds=rounds, games_per_round=8, iterations=25, epochs=20, seed=0,
            log=lambda r, lp, lv, n: print(f"round {r:2d} policy_loss={lp:.3f} value_loss={lv:.3f} buffer={n}") if r % 5 == 0 else None)
print(f"trained in {time.time()-t:.0f}s")
N = 40
def az(seed): return PUCT(g, iters, prior_fn=net.prior_fn, value_fn=net.value_fn, seed=seed)
print("AZ-lite  vs random :", match(g, az(1), random_player(g, 1), N))
print("AZ-lite  vs UCT    :", match(g, az(1), MCTS(g, iters, seed=2), N))
print("AZ-lite  vs perfect:", match(g, az(1), perfect_player(g, 1), N))
print("UCT      vs perfect:", match(g, MCTS(g, iters, seed=3), perfect_player(g, 1), N))
print("PUCT-uniform+rollout vs perfect:", match(g, PUCT(g, iters, seed=3), perfect_player(g, 1), N))
