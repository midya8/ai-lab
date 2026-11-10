"""Moved-code detection.

A block of lines removed in one place and added verbatim (modulo indentation)
elsewhere in the same diff is a *move*, not new code. Reviewing it as new code
produces noise: the findings already existed before the change. ``detect_moves``
finds such blocks so the reviewer can skip them.

Method: normalise each line (strip whitespace), drop trivial lines (blank, lone
brackets, comments-only) as anchors, then greedily match the longest runs of
consecutive removed lines against runs of consecutive added lines. A run must be
at least ``min_lines`` meaningful lines long to count, so repeated boilerplate
like ``return None`` does not create false moves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .diff import FileDiff

TRIVIAL = {"", "{", "}", "(", ")", "[", "]", "),", "],", "},", "else:", "pass", "end", "return", "break", "continue"}


@dataclass(frozen=True)
class Move:
    from_path: str
    from_line: int   # old-file line of the first moved line
    to_path: str
    to_line: int     # new-file line of the first moved line
    length: int      # number of lines in the run (including trivial ones inside it)


def _norm(text: str) -> str:
    return " ".join(text.split())


def _meaningful(norm: str) -> bool:
    return norm not in TRIVIAL and not norm.startswith(("#", "//"))


def _runs(kind: str, files: List[FileDiff]):
    """Yield (path, [(line_no, norm), ...]) for each maximal run of consecutive + or - lines."""
    for fd in files:
        for h in fd.hunks:
            run: List[Tuple[int, str]] = []
            for l in h.lines:
                if l.kind == kind:
                    no = l.new_no if kind == "+" else l.old_no
                    run.append((no, _norm(l.text)))
                else:
                    if run:
                        yield (fd.old_path or fd.path) if kind == "-" else fd.path, run
                    run = []
            if run:
                yield (fd.old_path or fd.path) if kind == "-" else fd.path, run


def detect_moves(files: List[FileDiff], min_lines: int = 3) -> List[Move]:
    removed = list(_runs("-", files))
    added = list(_runs("+", files))
    # index removed lines: norm -> [(run_idx, pos)]
    index: Dict[str, List[Tuple[int, int]]] = {}
    for ri, (_, run) in enumerate(removed):
        for pos, (_, n) in enumerate(run):
            index.setdefault(n, []).append((ri, pos))

    used_rem: Set[Tuple[int, int]] = set()
    moves: List[Move] = []
    for ai, (apath, arun) in enumerate(added):
        i = 0
        while i < len(arun):
            best = (0, None)  # (length, (ri, rpos))
            n = arun[i][1]
            if _meaningful(n):
                for ri, rpos in index.get(n, ()):
                    rrun = removed[ri][1]
                    k = 0
                    while (i + k < len(arun) and rpos + k < len(rrun) and arun[i + k][1] == rrun[rpos + k][1]
                           and (ri, rpos + k) not in used_rem):
                        k += 1
                    if k > best[0]:
                        best = (k, (ri, rpos))
            k = best[0]
            if best[1] is not None:
                ri, rpos = best[1]
                rrun = removed[ri][1]
                meaningful = sum(_meaningful(arun[i + j][1]) for j in range(k))
                if meaningful >= min_lines:
                    for j in range(k):
                        used_rem.add((ri, rpos + j))
                    moves.append(Move(removed[ri][0], rrun[rpos][0], apath, arun[i][0], k))
                    i += k
                    continue
            i += 1
    return moves


def moved_new_lines(moves: List[Move]) -> Dict[str, Set[int]]:
    """path -> set of new-file line numbers that are part of a move."""
    out: Dict[str, Set[int]] = {}
    for m in moves:
        out.setdefault(m.to_path, set()).update(range(m.to_line, m.to_line + m.length))
    return out
