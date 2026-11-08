"""Orchestration: diff -> rule findings (+ optional LLM findings) -> deduped, capped review."""
from __future__ import annotations

import fnmatch
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .diff import FileDiff, parse_diff
from .llm import Provider, build_prompt, chunk_diff, parse_reply, SYSTEM_PROMPT
from .rules import (Finding, RULE_TABLE, SEVERITIES, analyze_file, is_test_path, language_of,
                    sev_rank, suppressed)


@dataclass
class Config:
    disabled_rules: Set[str] = field(default_factory=set)
    severity_overrides: Dict[str, str] = field(default_factory=dict)
    ignore_paths: List[str] = field(default_factory=list)
    max_comments: int = 25
    min_severity: str = "low"
    complexity_threshold: int = 10
    max_file_bytes: int = 400_000

    def validate(self) -> "Config":
        if self.min_severity not in SEVERITIES:
            raise ValueError(f"min_severity must be one of {SEVERITIES}")
        for r, s in self.severity_overrides.items():
            if s not in SEVERITIES:
                raise ValueError(f"bad severity {s!r} for rule {r}")
        if self.max_comments < 1:
            raise ValueError("max_comments must be >= 1")
        return self


def load_config(path: Optional[str]) -> Config:
    if not path or not os.path.exists(path):
        return Config()
    with open(path) as f:
        raw = json.load(f)
    unknown = set(raw) - {"disabled_rules", "severity_overrides", "ignore_paths", "max_comments", "min_severity", "complexity_threshold"}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    cfg = Config(
        disabled_rules=set(raw.get("disabled_rules", [])),
        severity_overrides=dict(raw.get("severity_overrides", {})),
        ignore_paths=list(raw.get("ignore_paths", [])),
        max_comments=int(raw.get("max_comments", 25)),
        min_severity=raw.get("min_severity", "low"),
        complexity_threshold=int(raw.get("complexity_threshold", 10)),
    )
    return cfg.validate()


@dataclass
class ReviewResult:
    all_findings: List[Finding]          # everything that survived filtering, sorted
    findings: List[Finding]              # the capped list that will be commented
    omitted: int = 0
    suppressed: int = 0
    files_reviewed: int = 0
    files_skipped: int = 0
    notes: List[str] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)

    def max_severity(self) -> Optional[str]:
        return max((f.severity for f in self.all_findings), key=sev_rank, default=None)

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for f in self.all_findings:
            c[f.severity] = c.get(f.severity, 0) + 1
        return c

    def gate(self, fail_on: Optional[str]) -> bool:
        """True when the review should fail the build."""
        if not fail_on:
            return False
        return any(sev_rank(f.severity) >= sev_rank(fail_on) for f in self.all_findings)


_STOP = {"the", "a", "an", "is", "are", "to", "of", "and", "or", "in", "it", "this", "that", "be", "can", "will", "with", "for", "on", "as", "by", "if", "not", "may", "from"}


def _words(s: str) -> Set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", s.lower()) if w not in _STOP}


def _similar(a: str, b: str, threshold: float = 0.2) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


class Reviewer:
    def __init__(self, config: Optional[Config] = None, provider: Optional[Provider] = None, root: Optional[str] = None):
        self.config = (config or Config()).validate()
        self.provider = provider
        self.root = root

    def _read_source(self, fd: FileDiff, notes: List[str]) -> Optional[str]:
        if self.root is None or language_of(fd.path) != "python" or fd.status == "deleted":
            return None
        full = os.path.join(self.root, fd.path)
        try:
            if os.path.getsize(full) > self.config.max_file_bytes:
                notes.append(f"{fd.path}: larger than {self.config.max_file_bytes} bytes, AST rules skipped.")
                return None
            with open(full, encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            notes.append(f"{fd.path}: could not read from {self.root}, AST rules skipped.")
            return None

    def review(self, diff_text: str) -> ReviewResult:
        cfg = self.config
        files = parse_diff(diff_text)
        notes: List[str] = []
        stats = {"llm_calls": 0, "llm_dropped": 0, "llm_failed": 0, "llm_deduped": 0}
        found: List[Finding] = []
        n_supp = reviewed = skipped = 0
        src_changed = test_changed = False

        for fd in files:
            if any(fnmatch.fnmatch(fd.path, pat) for pat in cfg.ignore_paths) or fd.binary or fd.status == "deleted":
                skipped += 1
                continue
            reviewed += 1
            if language_of(fd.path):
                if is_test_path(fd.path):
                    test_changed = True
                else:
                    src_changed = True
            source = self._read_source(fd, notes)
            rule_f, file_notes = analyze_file(fd, source, cfg.complexity_threshold)
            notes.extend(file_notes)
            file_found = list(rule_f)

            if self.provider is not None and fd.added():
                added_map = fd.added()
                for chunk in chunk_diff(fd):
                    stats["llm_calls"] += 1
                    try:
                        reply = self.provider.complete(SYSTEM_PROMPT, build_prompt(chunk))
                    except Exception as e:  # a flaky model must not sink the whole review
                        stats["llm_failed"] += 1
                        notes.append(f"{fd.path}: LLM pass failed ({type(e).__name__}: {str(e)[:100]}).")
                        continue
                    llm_f, dropped = parse_reply(reply, fd.path, {n: added_map[n] for n in chunk.added})
                    stats["llm_dropped"] += dropped
                    for lf in llm_f:
                        dup = any(rf.line == lf.line and _similar(rf.message, lf.message) for rf in rule_f) or \
                            any(x.source == "llm" and x.line == lf.line and x.message == lf.message for x in file_found)
                        if dup:
                            stats["llm_deduped"] += 1
                        else:
                            file_found.append(lf)

            for f in file_found:
                if f.rule in cfg.disabled_rules:
                    continue
                if suppressed(f.line_text, f.rule):
                    n_supp += 1
                    continue
                sev = cfg.severity_overrides.get(f.rule, f.severity)
                if sev != f.severity:
                    f = Finding(f.path, f.line, f.rule, sev, f.message, f.suggestion, f.source, f.line_text)
                if sev_rank(f.severity) < sev_rank(cfg.min_severity):
                    continue
                found.append(f)

        found.sort(key=lambda f: (-sev_rank(f.severity), f.path, f.line, f.rule))
        capped = found[: cfg.max_comments]
        if src_changed and not test_changed:
            notes.append("Source files changed but no test files were touched.")
        if len(found) > len(capped):
            notes.append(f"{len(found) - len(capped)} lower-priority finding(s) not shown (max_comments={cfg.max_comments}).")
        return ReviewResult(found, capped, len(found) - len(capped), n_supp, reviewed, skipped, notes, stats)


# --------------------------------------------------------------------------- rendering

BADGE = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW", "info": "INFO"}
MARK = "reviewkit:fp="


def render_line_comment(findings: List[Finding]) -> str:
    """One inline comment for every finding that shares a (path, line)."""
    parts = []
    for f in findings:
        tag = f"**{BADGE[f.severity]}**" + (f" `{f.rule}`" if f.source == "rule" else " (model)")
        s = f"{tag}: {f.message}"
        if f.suggestion:
            s += f"\n\nSuggestion: {f.suggestion}"
        parts.append(s)
    fps = ",".join(f.fp for f in findings)
    return "\n\n---\n\n".join(parts) + f"\n\n<!-- {MARK}{fps} -->"


def group_by_line(findings: List[Finding]) -> Dict[tuple, List[Finding]]:
    g: Dict[tuple, List[Finding]] = defaultdict(list)
    for f in findings:
        g[(f.path, f.line)].append(f)
    return g


def render_summary(result: ReviewResult) -> str:
    c = result.counts()
    if not result.all_findings:
        head = f"reviewkit reviewed {result.files_reviewed} file(s) and found nothing to flag."
    else:
        bits = ", ".join(f"{c[s]} {s}" for s in reversed(SEVERITIES) if s in c)
        head = f"reviewkit reviewed {result.files_reviewed} file(s): {bits}."
    lines = [f"### {head}"]
    if result.findings:
        lines += ["", "| Severity | Location | Rule | Issue |", "|---|---|---|---|"]
        for f in result.findings:
            msg = f.message.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {f.severity} | `{f.path}:{f.line}` | {f.rule if f.source == 'rule' else 'model'} | {msg[:110]} |")
    if result.suppressed:
        lines += ["", f"{result.suppressed} finding(s) suppressed by `reviewkit: ignore` comments."]
    for n in result.notes:
        lines.append(f"- {n}")
    return "\n".join(lines)
