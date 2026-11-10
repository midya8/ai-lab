"""Re-test wide draft trees on a PEAKY target (follow-up to experiment_cost's null result).

Target: random first-order chain with Dirichlet(0.15) rows (concentrated). Drafts:
(a) tempered target (same ranking, flatter) and (b) a different random chain
mixed in. Cells = cost/token at c=0.1 (autoregressive = 1.0); * = best m."""
import numpy as np
from specdec import MarkovLM, TemperedLM, MixtureLM, tree_generate, tree_cost_per_token

V, N, SEEDS, DEPTH, C = 16, 200, 3, 4, 0.1
rng = np.random.default_rng(0)
target = MarkovLM(rng.dirichlet(np.full(V, 0.15), size=V))
other = MarkovLM(rng.dirichlet(np.full(V, 0.15), size=V))
ent = np.mean([-(r * np.log(r + 1e-12)).sum() for r in target.trans])
print(f"target mean row entropy {ent:.2f} nats (uniform would be {np.log(V):.2f})")
drafts = {f"temper T={T}": TemperedLM(target, T) for T in (1.5, 3.0, 6.0)}
drafts.update({f"mix w={w}": MixtureLM(target, other, w) for w in (0.9, 0.6, 0.3)})
print(f"cost/token, c={C}, depth {DEPTH}, m in (1,2,4,8)")
for name, draft in drafts.items():
    row, acc = [], []
    for m in (1, 2, 4, 8):
        tot = 0.0
        for s in range(SEEDS):
            st = {}
            tree_generate(target, draft, [0], N, DEPTH, m, np.random.default_rng(s), st)
            tot += tree_cost_per_token(st, N, C)
        row.append(tot / SEEDS)
    b = int(np.argmin(row))
    print(f"  {name:12s} | " + " ".join(f"{v:.3f}{'*' if i == b else ' '}" for i, v in enumerate(row)))
