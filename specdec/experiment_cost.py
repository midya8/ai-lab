"""Cost-charged version of experiment_tree: which m minimises cost per token?
cost = (target_calls + c * draft_nodes) / tokens, autoregressive = 1.0."""
import numpy as np
from specdec import NgramLM, MixtureLM, tree_generate, tree_cost_per_token

rng = np.random.default_rng(0)
toks = list(rng.integers(0, 4, 3000))
target = NgramLM(4, 2, 0.1).fit(toks)
other = NgramLM(4, 1, 1.0).fit(toks[::-1])
N, SEEDS, DEPTH = 200, 3, 4
print("draft quality = w_target; cells = cost/token (lower better); * = best m")
for c in (0.02, 0.1, 0.3):
    print(f"\ncost_ratio c={c}")
    for w in (0.2, 0.6, 0.9):
        draft = MixtureLM(target, other, w)
        row = []
        for m in (1, 2, 4, 8):
            tot = 0.0
            for s in range(SEEDS):
                st = {}
                tree_generate(target, draft, [0], N, DEPTH, m, np.random.default_rng(s), st)
                tot += tree_cost_per_token(st, N, c)
            row.append(tot / SEEDS)
        b = int(np.argmin(row))
        print(f"  w={w:.1f} | " + " ".join(f"{v:.3f}{'*' if i == b else ' '}" for i, v in enumerate(row)))
