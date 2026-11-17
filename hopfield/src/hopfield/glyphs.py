"""Ten 5x7 digit bitmaps: small, structured, *correlated* patterns (unlike i.i.d. random ones)."""
from __future__ import annotations

import numpy as np

_RAW = {
    0: [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    1: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    2: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    3: ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    4: ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    5: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    6: ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    7: ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    8: [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    9: [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
}
ROWS, COLS = 7, 5

DIGITS = {k: np.array([[1.0 if c == "#" else -1.0 for c in row] for row in v]) for k, v in _RAW.items()}


def digit_patterns(which=range(10)) -> np.ndarray:
    """Flattened +-1 patterns, shape (len(which), 35)."""
    return np.stack([DIGITS[d].ravel() for d in which])


def render(x: np.ndarray, on: str = "#", off: str = ".") -> str:
    """ASCII picture of a flattened 35-vector (sign decides on/off)."""
    g = np.asarray(x).reshape(ROWS, COLS)
    return "\n".join("".join(on if v > 0 else off for v in row) for row in g)
