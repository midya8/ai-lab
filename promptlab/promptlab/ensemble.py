"""Multi-judge pairwise ensembles with inter-judge agreement.

Each judge rules on every pair in both orders (``judge_pair``); the ensemble takes a
strict-majority vote over the judges' reconciled winners and reports how much the
judges agree (raw agreement, unanimity, Fleiss' kappa), so one noisy judge cannot
silently decide an A/B result.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .pairwise import PairJudge, PairVerdict, PairwiseSummary, judge_pair, summarize

LABELS = ("a", "b", "tie")


def fleiss_kappa(rows: list[list[str]]) -> float | None:
    """Fleiss' kappa over items rated by the same number (>= 2) of raters; ``None`` if undefined."""
    if not rows:
        raise ValueError("no items")
    m = len(rows[0])
    if m < 2 or any(len(r) != m for r in rows):
        raise ValueError("every item needs the same number (>= 2) of ratings")
    n = len(rows)
    p_bar = 0.0
    totals = Counter()
    for r in rows:
        c = Counter(r)
        totals.update(c)
        p_bar += (sum(v * v for v in c.values()) - m) / (m * (m - 1))
    p_bar /= n
    pe = sum((totals[l] / (n * m)) ** 2 for l in LABELS)
    if abs(1 - pe) < 1e-12:
        return None
    return (p_bar - pe) / (1 - pe)


@dataclass
class EnsembleVerdict:
    id: str
    winner: str                      # "a" | "b" | "tie"
    votes: dict[str, str]            # judge name -> "a" | "b" | "tie" | "error"
    unanimous: bool
    kind: str                        # "majority" | "split" | "error"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EnsembleResult:
    names: list[str]
    verdicts: list[EnsembleVerdict]
    per_judge: dict[str, PairwiseSummary]
    summary: PairwiseSummary
    kappa: float | None
    agreement: float                 # mean pairwise raw agreement over fully-voted items
    unanimity: float                 # share of fully-voted items all judges decided identically
    full_items: int = 0
    skipped: int = 0
    _extra: dict = field(default_factory=dict)


def vote(votes: dict[str, str], quorum: int) -> tuple[str, str]:
    """Strict majority of the valid votes; fewer than ``quorum`` valid votes is an error."""
    valid = [v for v in votes.values() if v != "error"]
    if len(valid) < quorum:
        return "tie", "error"
    top, n = Counter(valid).most_common(1)[0]
    if n * 2 > len(valid):
        return top, "majority"
    return "tie", "split"


def compare_ensemble(judges: dict[str, PairJudge], items: list[dict], alpha: float = 0.05,
                     quorum: int | None = None) -> EnsembleResult:
    if len(judges) < 2:
        raise ValueError("an ensemble needs at least two judges")
    names = list(judges)
    quorum = quorum or (len(names) // 2 + 1)
    per: dict[str, list[PairVerdict]] = {n: [] for n in names}
    out, rows, skipped = [], [], 0
    for it in items:
        if not (it["a"] or "").strip() or not (it["b"] or "").strip():
            skipped += 1
            continue
        votes = {}
        for n in names:
            pv = judge_pair(judges[n], it["rubric"], it["a"], it["b"], it.get("vars"), str(it["id"]))
            per[n].append(pv)
            votes[n] = "error" if pv.kind == "error" else pv.winner
        winner, kind = vote(votes, quorum)
        valid = [v for v in votes.values() if v != "error"]
        out.append(EnsembleVerdict(str(it["id"]), winner, votes, len(set(valid)) == 1 and len(valid) == len(names), kind))
        if len(valid) == len(names):
            rows.append([votes[n] for n in names])
    pvs = []
    for ev in out:
        if ev.kind == "error":
            pvs.append(PairVerdict(ev.id, "tie", "error", error="too few judges produced a verdict"))
        else:
            pvs.append(PairVerdict(ev.id, ev.winner, "agree" if ev.kind == "majority" else "hesitant"))
    k = fleiss_kappa(rows) if rows else None
    pairs = len(names) * (len(names) - 1) / 2
    agree = (sum(sum(c * (c - 1) / 2 for c in Counter(r).values()) / pairs for r in rows) / len(rows)) if rows else 0.0
    unan = sum(len(set(r)) == 1 for r in rows) / len(rows) if rows else 0.0
    return EnsembleResult(names, out, {n: summarize(v, skipped, alpha) for n, v in per.items()},
                          summarize(pvs, skipped, alpha), k, agree, unan, len(rows), skipped)


def render_ensemble(r: EnsembleResult, name_a: str = "A", name_b: str = "B") -> str:
    s = r.summary
    lines = [f"Ensemble of {len(r.names)} judges ({', '.join(r.names)}), {s.n} pair(s) ({name_a} = A, {name_b} = B)"]
    lines.append(f"  ensemble: {name_a} wins {s.wins_a}, {name_b} wins {s.wins_b}, ties/splits {s.ties}, "
                 f"undecidable {s.errors}  ->  {s.verdict}")
    lines.append("  per judge (A wins / B wins / ties / flips / errors):")
    for n in r.names:
        j = r.per_judge[n]
        lines.append(f"    {n:<14} {j.wins_a:>3} / {j.wins_b:>3} / {j.ties:>3} / {j.position + j.hesitant:>3} / {j.errors:>2}"
                     f"   -> {j.verdict}")
    if r.full_items:
        ks = "undefined (no variation)" if r.kappa is None else f"{r.kappa:.2f}"
        lines.append(f"  agreement over {r.full_items} fully-judged pair(s): raw {r.agreement:.1%}, "
                     f"unanimous {r.unanimity:.1%}, Fleiss kappa {ks}")
        if r.kappa is not None and r.kappa < 0.4:
            lines.append("  WARNING: low inter-judge agreement; the ensemble result is weak evidence.")
    return "\n".join(lines)
