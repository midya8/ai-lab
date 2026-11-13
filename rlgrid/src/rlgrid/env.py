"""Deterministic/slippery gridworld with walls, pits and a goal."""
import numpy as np

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
DEFAULT_MAP = [
    "S..#....",
    ".#.#.##.",
    ".#...#..",
    ".####.#.",
    "...P..#G",
]


class GridWorld:
    """Map chars: S start, G goal (+1), P pit (-1, terminal), # wall, . floor.
    Step cost -0.01. With prob `slip` the action is replaced by a random one."""

    def __init__(self, layout=None, slip=0.0, step_cost=-0.01, seed=0):
        self.layout = layout or DEFAULT_MAP
        self.h, self.w = len(self.layout), len(self.layout[0])
        assert all(len(r) == self.w for r in self.layout), "ragged map"
        self.slip, self.step_cost = slip, step_cost
        self.rng = np.random.default_rng(seed)
        cells = {c: [] for c in "SGP#."}
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                cells[ch].append((r, c))
        assert len(cells["S"]) == 1 and len(cells["G"]) >= 1, "need one S and a G"
        self.start = cells["S"][0]
        self.goals, self.pits, self.walls = set(cells["G"]), set(cells["P"]), set(cells["#"])
        self.n_states, self.n_actions = self.h * self.w, 4
        self.pos = self.start

    def sid(self, pos):
        return pos[0] * self.w + pos[1]

    def terminal(self, pos):
        return pos in self.goals or pos in self.pits

    def _move(self, pos, a):
        r, c = pos[0] + ACTIONS[a][0], pos[1] + ACTIONS[a][1]
        if not (0 <= r < self.h and 0 <= c < self.w) or (r, c) in self.walls:
            return pos
        return (r, c)

    def reward(self, nxt):
        return 1.0 if nxt in self.goals else -1.0 if nxt in self.pits else self.step_cost

    def reset(self):
        self.pos = self.start
        return self.sid(self.pos)

    def step(self, a):
        if self.slip and self.rng.random() < self.slip:
            a = int(self.rng.integers(4))
        self.pos = self._move(self.pos, a)
        return self.sid(self.pos), self.reward(self.pos), self.terminal(self.pos)

    def transitions(self, pos, a):
        """[(prob, next_pos)] under slip, for planning."""
        out = {}
        for b in range(4):
            p = (1 - self.slip if b == a else 0.0) + self.slip / 4
            if p:
                n = self._move(pos, b)
                out[n] = out.get(n, 0) + p
        return [(p, n) for n, p in out.items()]

    def free_cells(self):
        return [(r, c) for r in range(self.h) for c in range(self.w)
                if (r, c) not in self.walls and not self.terminal((r, c))]

    def render(self, policy=None):
        arrows = "^v<>"
        rows = []
        for r in range(self.h):
            line = ""
            for c in range(self.w):
                ch = self.layout[r][c]
                if policy is not None and ch in ".S":
                    ch = arrows[policy[self.sid((r, c))]]
                line += ch
            rows.append(line)
        return "\n".join(rows)
