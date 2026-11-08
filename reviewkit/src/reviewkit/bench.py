"""Benchmark harness: labelled before/after files -> real unified diffs -> precision/recall per rule.

Each case is {name, files: {path: (before_or_None, after)}}. Expectations live in the code
as `# EXPECT: rule-id[, rule-id]` (or `// EXPECT:`) on the line that should be flagged.
Lines without a marker must produce no finding, so negatives and "risky code that was
already there" cases are measured too.
"""
from __future__ import annotations

import difflib
import importlib.util
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .review import Config, Reviewer

EXPECT_RE = re.compile(r"(?:#|//)\s*EXPECT:\s*([\w\-]+(?:\s*,\s*[\w\-]+)*)")


@dataclass
class BenchResult:
    tp: Dict[str, int]
    fp: Dict[str, int]
    fn: Dict[str, int]
    false_positives: List[str]
    missed: List[str]
    cases: int

    def totals(self) -> Tuple[int, int, int]:
        return sum(self.tp.values()), sum(self.fp.values()), sum(self.fn.values())

    def precision(self) -> float:
        tp, fp, _ = self.totals()
        return tp / (tp + fp) if tp + fp else 1.0

    def recall(self) -> float:
        tp, _, fn = self.totals()
        return tp / (tp + fn) if tp + fn else 1.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if p + r else 0.0


def load_cases(path: str):
    spec = importlib.util.spec_from_file_location("bench_cases", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES


def make_diff(files: Dict[str, Tuple[Optional[str], str]]) -> str:
    out: List[str] = []
    for path, (before, after) in files.items():
        a = before.split("\n") if before is not None else []
        b = after.split("\n")
        fromfile = f"a/{path}" if before is not None else "/dev/null"
        out.extend(difflib.unified_diff(a, b, fromfile=fromfile, tofile=f"b/{path}", n=3, lineterm=""))
    return "\n".join(out) + "\n"


def expected_for(files: Dict[str, Tuple[Optional[str], str]]) -> Set[Tuple[str, int, str]]:
    exp = set()
    for path, (before, after) in files.items():
        before_lines = set((before or "").split("\n"))
        for i, line in enumerate(after.split("\n"), 1):
            m = EXPECT_RE.search(line)
            if m:
                assert line not in before_lines, f"{path}:{i} EXPECT marker on a line that is not new"
                for r in m.group(1).split(","):
                    exp.add((path, i, r.strip()))
    return exp


def run_bench(cases, config: Optional[Config] = None) -> BenchResult:
    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)
    false_pos: List[str] = []
    missed: List[str] = []
    for case in cases:
        files = case["files"]
        exp = expected_for(files)
        with tempfile.TemporaryDirectory() as root:
            for path, (_, after) in files.items():
                full = os.path.join(root, path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(after)
            res = Reviewer(config or Config(max_comments=10_000, min_severity="info"), root=root).review(make_diff(files))
        got = {(f.path, f.line, f.rule) for f in res.all_findings}
        for k in got & exp:
            tp[k[2]] += 1
        for k in got - exp:
            fp[k[2]] += 1
            false_pos.append(f"{case['name']}: {k[0]}:{k[1]} {k[2]}")
        for k in exp - got:
            fn[k[2]] += 1
            missed.append(f"{case['name']}: {k[0]}:{k[1]} {k[2]}")
    return BenchResult(dict(tp), dict(fp), dict(fn), false_pos, missed, len(cases))


def format_bench(r: BenchResult) -> str:
    rules = sorted(set(r.tp) | set(r.fp) | set(r.fn))
    lines = [f"{r.cases} cases", "", f"{'rule':26} {'TP':>3} {'FP':>3} {'FN':>3}"]
    for rule in rules:
        lines.append(f"{rule:26} {r.tp.get(rule, 0):>3} {r.fp.get(rule, 0):>3} {r.fn.get(rule, 0):>3}")
    tp, fp, fn = r.totals()
    lines += ["", f"TP {tp}  FP {fp}  FN {fn}   precision {r.precision():.3f}  recall {r.recall():.3f}  F1 {r.f1():.3f}"]
    if r.false_positives:
        lines += ["", "False positives:"] + [f"  {x}" for x in r.false_positives]
    if r.missed:
        lines += ["", "Missed:"] + [f"  {x}" for x in r.missed]
    return "\n".join(lines)
