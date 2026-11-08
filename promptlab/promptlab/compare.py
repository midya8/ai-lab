"""Paired A/B comparison of two runs over the same cases, with a verdict."""
from __future__ import annotations

from dataclasses import dataclass, field

from .runner import RunResult
from .stats import mcnemar_exact, paired_bootstrap, wilson


@dataclass
class Comparison:
    a: RunResult
    b: RunResult
    n: int
    a_ci: tuple[float, float]
    b_ci: tuple[float, float]
    only_a: list[str]          # case ids A passed and B failed
    only_b: list[str]          # case ids B passed and A failed
    p_value: float
    score_diff: float
    score_ci: tuple[float, float]
    verdict: str
    alpha: float = 0.05
    notes: list[str] = field(default_factory=list)


def compare_runs(a: RunResult, b: RunResult, alpha: float = 0.05, seed: int = 0) -> Comparison:
    """Compare on the intersection of case ids; cases missing on one side are reported, not guessed."""
    ra, rb = a.by_id(), b.by_id()
    ids = [r.id for r in a.results if r.id in rb]
    notes = []
    skipped = (set(ra) ^ set(rb))
    if skipped:
        notes.append(f"{len(skipped)} case(s) present in only one run were ignored")
    if not ids:
        raise ValueError("the two runs share no case ids")
    only_a = [i for i in ids if ra[i].passed and not rb[i].passed]
    only_b = [i for i in ids if rb[i].passed and not ra[i].passed]
    p = mcnemar_exact(len(only_a), len(only_b))
    ka = sum(ra[i].passed for i in ids)
    kb = sum(rb[i].passed for i in ids)
    diff, lo, hi = paired_bootstrap([ra[i].score for i in ids], [rb[i].score for i in ids], seed=seed)
    if len(only_a) + len(only_b) == 0:
        verdict = "identical pass/fail on every case"
    elif p < alpha and len(only_b) > len(only_a):
        verdict = f"B is better (p={p:.4f})"
    elif p < alpha and len(only_a) > len(only_b):
        verdict = f"A is better (p={p:.4f})"
    else:
        verdict = f"no significant difference (p={p:.4f})"
        notes.append(f"only {len(only_a) + len(only_b)} case(s) differ; add cases to detect a smaller effect")
    return Comparison(a, b, len(ids), wilson(ka, len(ids)), wilson(kb, len(ids)),
                      only_a, only_b, p, diff, (lo, hi), verdict, alpha, notes)


def render_markdown(c: Comparison) -> str:
    ka = sum(c.a.by_id()[i].passed for i in c.a.by_id() if i in c.b.by_id())
    kb = sum(c.b.by_id()[i].passed for i in c.b.by_id() if i in c.a.by_id())
    lines = [
        f"# A/B: {c.a.prompt} v{c.a.version} vs v{c.b.version}", "",
        f"**Verdict: {c.verdict}**", "",
        "| | A (v%d) | B (v%d) |" % (c.a.version, c.b.version), "|---|---|---|",
        f"| pass rate | {ka}/{c.n} = {ka / c.n:.1%} | {kb}/{c.n} = {kb / c.n:.1%} |",
        f"| 95% CI (Wilson) | {c.a_ci[0]:.1%} – {c.a_ci[1]:.1%} | {c.b_ci[0]:.1%} – {c.b_ci[1]:.1%} |",
        f"| avg prompt words | {c.a.prompt_words:.0f} | {c.b.prompt_words:.0f} |",
        f"| avg output words | {c.a.output_words:.0f} | {c.b.output_words:.0f} |", "",
        f"Mean score change (B − A): {c.score_diff:+.3f}, 95% bootstrap CI [{c.score_ci[0]:+.3f}, {c.score_ci[1]:+.3f}]",
        f"Paired outcomes: {len(c.only_b)} fixed by B, {len(c.only_a)} broken by B (exact McNemar p = {c.p_value:.4f}).",
    ]
    if c.only_a:
        lines += ["", "Regressions (A passed, B failed): " + ", ".join(c.only_a)]
    if c.only_b:
        lines += ["", "Improvements (B passed, A failed): " + ", ".join(c.only_b)]
    for n in c.notes:
        lines += ["", f"_Note: {n}_"]
    return "\n".join(lines) + "\n"


@dataclass
class RegressionReport:
    regressions: list[str]
    fixed: list[str]
    new_cases: list[str]
    missing_cases: list[str]
    pass_rate: float
    baseline_pass_rate: float
    ok: bool
    reasons: list[str]


def check_regression(current: RunResult, baseline: RunResult, max_regressions: int = 0,
                     min_pass_rate: float | None = None, max_pass_rate_drop: float | None = None) -> RegressionReport:
    """Gate a candidate prompt against a saved baseline run.

    A *regression* is a case that passed in the baseline and fails now. This
    is stricter than comparing pass rates: a prompt can raise the average
    while quietly breaking cases that used to work.
    """
    cur, base = current.by_id(), baseline.by_id()
    shared = [i for i in base if i in cur]
    regs = [i for i in shared if base[i].passed and not cur[i].passed]
    fixed = [i for i in shared if cur[i].passed and not base[i].passed]
    new = [i for i in cur if i not in base]
    missing = [i for i in base if i not in cur]
    reasons = []
    if len(regs) > max_regressions:
        reasons.append(f"{len(regs)} regression(s) > allowed {max_regressions}: {', '.join(regs)}")
    if min_pass_rate is not None and current.pass_rate < min_pass_rate:
        reasons.append(f"pass rate {current.pass_rate:.1%} < required {min_pass_rate:.1%}")
    if max_pass_rate_drop is not None and baseline.pass_rate - current.pass_rate > max_pass_rate_drop:
        reasons.append(f"pass rate dropped {baseline.pass_rate - current.pass_rate:.1%} > allowed {max_pass_rate_drop:.1%}")
    if missing:
        reasons.append(f"{len(missing)} baseline case(s) not run: {', '.join(missing)}")
    return RegressionReport(regs, fixed, new, missing, current.pass_rate, baseline.pass_rate, not reasons, reasons)


def render_check_markdown(rep: RegressionReport) -> str:
    """Markdown summary of a regression gate, suitable for $GITHUB_STEP_SUMMARY."""
    lines = [f"### Prompt regression gate: {'PASS' if rep.ok else 'FAIL'}", "",
             f"Pass rate **{rep.pass_rate:.1%}** (baseline {rep.baseline_pass_rate:.1%}). "
             f"{len(rep.regressions)} regressed, {len(rep.fixed)} fixed, {len(rep.new_cases)} new."]
    for title, ids in (("Regressed", rep.regressions), ("Fixed", rep.fixed), ("New", rep.new_cases)):
        if ids:
            lines += ["", f"**{title}:** " + ", ".join(f"`{i}`" for i in ids)]
    if rep.reasons:
        lines += ["", "**Why it failed:**"] + [f"- {r}" for r in rep.reasons]
    return "\n".join(lines) + "\n"
