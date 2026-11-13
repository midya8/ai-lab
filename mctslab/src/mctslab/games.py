"""Two-player zero-sum games with a tiny shared interface.

State is an immutable tuple board + player to move. Players are +1 / -1.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectN:
    """Gravity-free m x n board where first to get k in a row wins (k=3, 3x3 is tic-tac-toe).
    With gravity=True pieces drop (Connect-Four style)."""
    rows: int = 3
    cols: int = 3
    k: int = 3
    gravity: bool = False

    def initial(self):
        return (tuple([0] * (self.rows * self.cols)), 1)

    def legal_moves(self, state):
        board, _ = state
        if self.winner(state) != 0:
            return []
        if self.gravity:
            return [c for c in range(self.cols) if board[c] == 0]  # row 0 is top
        return [i for i, v in enumerate(board) if v == 0]

    def play(self, state, move):
        board, player = state
        if move not in self.legal_moves(state):
            raise ValueError(f"illegal move {move}")
        b = list(board)
        if self.gravity:
            r = max(r for r in range(self.rows) if b[r * self.cols + move] == 0)
            b[r * self.cols + move] = player
        else:
            b[move] = player
        return (tuple(b), -player)

    def winner(self, state):
        board, _ = state
        R, C, k = self.rows, self.cols, self.k
        for r in range(R):
            for c in range(C):
                p = board[r * C + c]
                if p == 0:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    er, ec = r + dr * (k - 1), c + dc * (k - 1)
                    if not (0 <= er < R and 0 <= ec < C):
                        continue
                    if all(board[(r + dr * i) * C + c + dc * i] == p for i in range(k)):
                        return p
        return 0

    def is_terminal(self, state):
        return self.winner(state) != 0 or all(v != 0 for v in state[0])

    def render(self, state):
        sym = {0: ".", 1: "X", -1: "O"}
        b = state[0]
        return "\n".join("".join(sym[b[r * self.cols + c]] for c in range(self.cols)) for r in range(self.rows))


def TicTacToe():
    return ConnectN(3, 3, 3, False)
