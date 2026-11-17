"""Root-parallel UCT: several independent trees, visit counts summed at the root.

Each worker gets its own seed and iterations // workers budget, so the total
simulation budget equals a single search. Workers can run in processes
(``processes=True``) or sequentially; results are identical either way.
"""
from __future__ import annotations
import random
from concurrent.futures import ProcessPoolExecutor
from .mcts import uct_search


def _one(args):
    game, state, iters, c, seed = args
    return uct_search(game, state, iters, c, random.Random(seed))[1]


def root_parallel_search(game, state, iterations=500, workers=4, c=1.4, seed=0, processes=False):
    """Return (best_move, merged_visit_counts)."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    moves = game.legal_moves(state)
    if not moves:
        raise ValueError("no legal moves")
    if len(moves) == 1:
        return moves[0], {moves[0]: iterations}
    base, extra = divmod(iterations, workers)
    jobs = [(game, state, base + (1 if i < extra else 0), c, seed * 1000 + i)
            for i in range(workers) if base + (1 if i < extra else 0) > 0]
    if processes:
        with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
            results = list(ex.map(_one, jobs))
    else:
        results = [_one(j) for j in jobs]
    merged = {}
    for counts in results:
        for m, n in counts.items():
            merged[m] = merged.get(m, 0) + n
    best = max(merged, key=lambda m: (merged[m], -m))
    return best, merged


class RootParallelMCTS:
    def __init__(self, game, iterations=500, workers=4, c=1.4, seed=0, processes=False):
        self.game, self.iterations, self.workers = game, iterations, workers
        self.c, self.processes = c, processes
        self.rng = random.Random(seed)

    def __call__(self, state):
        return root_parallel_search(self.game, state, self.iterations, self.workers, self.c,
                                    self.rng.randrange(10**6), self.processes)[0]
