"""Unified-diff parser. Tracks new-file line numbers so comments land on real lines."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class DiffLine:
    kind: str  # "+", "-" or " "
    text: str
    old_no: Optional[int]
    new_no: Optional[int]


@dataclass
class Hunk:
    header: str
    old_start: int
    new_start: int
    lines: List[DiffLine] = field(default_factory=list)


@dataclass
class FileDiff:
    path: str
    old_path: Optional[str] = None
    status: str = "modified"  # added | deleted | modified | renamed
    binary: bool = False
    hunks: List[Hunk] = field(default_factory=list)

    def added(self) -> Dict[int, str]:
        """New-file line number -> text, for every added line."""
        return {l.new_no: l.text for h in self.hunks for l in h.lines if l.kind == "+"}

    def visible(self) -> Dict[int, str]:
        """New-file lines shown in the diff (added + context)."""
        return {l.new_no: l.text for h in self.hunks for l in h.lines if l.kind in "+ "}


def _unquote(p: str) -> str:
    p = p.split("\t")[0].strip()
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        p = p[1:-1]
    return p


def _strip_ab(p: str) -> str:
    return p[2:] if p[:2] in ("a/", "b/") else p


def _git_header_path(line: str) -> str:
    rest = line[len("diff --git "):]
    idx = rest.rfind(" b/")
    if idx >= 0:
        return _unquote(rest[idx + 1:])[2:]
    return _strip_ab(_unquote(rest.split()[-1]))


def parse_diff(text: str) -> List[FileDiff]:
    lines = text.split("\n")
    files: List[FileDiff] = []
    cur: Optional[FileDiff] = None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            cur = FileDiff(path=_git_header_path(line))
            files.append(cur)
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            old, new = _unquote(line[4:]), _unquote(lines[i + 1][4:])
            if cur is None or cur.hunks:  # plain `diff -u` without a git header
                cur = FileDiff(path="")
                files.append(cur)
            if old == "/dev/null":
                cur.status = "added"
            elif new == "/dev/null":
                cur.status = "deleted"
            cur.old_path = None if old == "/dev/null" else _strip_ab(old)
            cur.path = _strip_ab(new) if new != "/dev/null" else _strip_ab(old)
            i += 2
            continue
        if cur is not None:
            if line.startswith("new file mode"):
                cur.status = "added"
            elif line.startswith("deleted file mode"):
                cur.status = "deleted"
            elif line.startswith("rename from "):
                cur.old_path, cur.status = line[12:].strip(), "renamed"
            elif line.startswith("rename to "):
                cur.path, cur.status = line[10:].strip(), "renamed"
            elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
                cur.binary = True
        m = HUNK_RE.match(line)
        if m and cur is not None:
            old_start, old_n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
            new_start, new_n = int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1)
            hunk = Hunk(header=line, old_start=old_start, new_start=new_start)
            o, nw = old_start, new_start
            i += 1
            while (old_n > 0 or new_n > 0) and i < n:
                l = lines[i]
                if l.startswith("\\"):
                    i += 1
                    continue
                c = l[:1]
                if c == "+":
                    hunk.lines.append(DiffLine("+", l[1:], None, nw))
                    nw += 1
                    new_n -= 1
                elif c == "-":
                    hunk.lines.append(DiffLine("-", l[1:], o, None))
                    o += 1
                    old_n -= 1
                elif c == " " or l == "":
                    hunk.lines.append(DiffLine(" ", l[1:], o, nw))
                    o += 1
                    nw += 1
                    old_n -= 1
                    new_n -= 1
                else:
                    break
                i += 1
            while i < n and lines[i].startswith("\\"):
                i += 1
            cur.hunks.append(hunk)
            continue
        i += 1
    return files
