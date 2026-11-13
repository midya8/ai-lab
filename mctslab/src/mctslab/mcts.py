"""UCT Monte-Carlo tree search with random rollouts."""
from __future__ import annotations
import math
import random


class Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "n", "w")

    def __init__(self, game, state, parent=None, move=None):
        self.state, self.parent, self.move = state, parent, move
        self.children = []
        self.untried = list(game.legal_moves(state))
        self.n = 0
        self.w = 0.0  # total reward for the player who moved INTO this node


def _rollout(game, state, rng):
    while not game.is_terminal(state):
        state = game.play(state, rng.choice(game.legal_moves(state)))
    return game.winner(state)  # +1/-1/0 absolute


def uct_search(game, state, iterations=500, c=1.4, rng=None):
    """Return (best_move, visit_counts dict). Move chosen by most visits."""
    rng = rng or random.Random(0)
    moves = game.legal_moves(state)
    if not moves:
        raise ValueError("no legal moves")
    if len(moves) == 1:
        return moves[0], {moves[0]: iterations}
    root = Node(game, state)
    for _ in range(iterations):
        node = root
        while not node.untried and node.children:
            log_n = math.log(node.n)
            node = max(node.children, key=lambda ch: ch.w / ch.n + c * math.sqrt(log_n / ch.n))
        if node.untried:
            m = node.untried.pop(rng.randrange(len(node.untried)))
            child = Node(game, game.play(node.state, m), node, m)
            node.children.append(child)
            node = child
        winner = _rollout(game, node.state, rng)
        while node is not None:
            mover = -node.state[1]  # player who moved into node
            if node.parent is not None:
                node.w += 1.0 if winner == mover else (0.5 if winner == 0 else 0.0)
            node.n += 1
            node = node.parent
    counts = {ch.move: ch.n for ch in root.children}
    best = max(counts, key=lambda m: (counts[m], -m))
    return best, counts


class MCTS:
    """Callable player: MCTS(game, iterations)(state) -> move."""

    def __init__(self, game, iterations=500, c=1.4, seed=0):
        self.game, self.iterations, self.c = game, iterations, c
        self.rng = random.Random(seed)

    def __call__(self, state):
        return uct_search(self.game, state, self.iterations, self.c, self.rng)[0]
