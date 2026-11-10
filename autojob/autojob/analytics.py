"""Funnel analytics over the application log.

Everything here is arithmetic on the user's own tracker entries — counts,
conversion rates, stage durations, and stale applications. No benchmarks,
no external data, no claims beyond what the log itself records.

Stage durations come only from the tracker's recorded history (v0.2+).
Entries logged before history existed still count toward the funnel, but
contribute nothing to durations: a move whose start date was never recorded
is not guessed at.
"""

from __future__ import annotations

import datetime
import json

FUNNEL_STAGES = ("applied", "screening", "interview", "offer", "hired")
TERMINAL_STATUSES = ("hired", "rejected", "withdrawn")


def _parse_date(value):
    """Parse an ISO date string; return None for anything unparseable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def history_of(entry):
    """Status history as a list of {"status", "at"} dicts.

    Entries written before v0.2 have no history: they contribute a single
    applied event dated at ``applied_at``. Anything later than that is
    unknown, so no later event is invented.
    """
    history = entry.get("history")
    if isinstance(history, list) and history:
        return [dict(h) for h in history if isinstance(h, dict)]
    applied_at = entry.get("applied_at")
    if applied_at:
        return [{"status": "applied", "at": applied_at}]
    return []


def reached_stages(entry):
    """Funnel stages this application is known to have reached."""
    reached = {h["status"] for h in history_of(entry) if h.get("status") in FUNNEL_STAGES}
    current = entry.get("status")
    if current in FUNNEL_STAGES:
        reached.add(current)
    return reached


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def funnel(entries):
    """Funnel counts and stage-to-stage conversion rates."""
    total = len(entries)
    reached = {stage: 0 for stage in FUNNEL_STAGES}
    for entry in entries:
        for stage in reached_stages(entry):
            reached[stage] += 1
    conversions = {}
    previous = None
    for stage in FUNNEL_STAGES:
        if previous is None:
            conversions[stage] = None
        else:
            base = reached[previous]
            conversions[stage] = (reached[stage] / base) if base else None
        previous = stage
    outcomes = {}
    for entry in entries:
        status = entry.get("status")
        outcomes[status] = outcomes.get(status, 0) + 1
    active = sum(1 for e in entries if e.get("status") not in TERMINAL_STATUSES)
    return {
        "total": total,
        "reached": reached,
        "conversions": conversions,
        "outcomes": outcomes,
        "active": active,
        "response_rate": (reached["screening"] / total) if total else None,
        "rejection_rate": (outcomes.get("rejected", 0) / total) if total else None,
    }


def days_in_stage(entries):
    """Median days spent in each stage, over completed stays only.

    A stay is the gap between two consecutive recorded history events.
    The current (in-progress) stay is excluded — it has no end date yet.
    Returns {stage: {"median_days": float, "n": int}} for stages with data.
    """
    stays = {}
    for entry in entries:
        # Only status events delimit a stay. A follow-up recorded mid-stage
        # is activity, not a stage change — pairing across it would split
        # one stay into two and undercount it.
        history = [h for h in history_of(entry) if h.get("status")]
        for earlier, later in zip(history, history[1:]):
            start = _parse_date(earlier.get("at"))
            end = _parse_date(later.get("at"))
            stage = earlier.get("status")
            if start is None or end is None or stage not in FUNNEL_STAGES:
                continue
            days = (end - start).days
            if days < 0:
                continue
            stays.setdefault(stage, []).append(days)
    return {
        stage: {"median_days": _median(days), "n": len(days)}
        for stage, days in sorted(stays.items())
        if days
    }


def stale_applications(entries, stale_days, today):
    """Non-terminal applications with no movement for over ``stale_days``.

    Movement means the latest recorded history event (or ``applied_at``
    when there is no history). An application moved exactly ``stale_days``
    ago is not stale yet.
    """
    stale = []
    for entry in entries:
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        last = None
        for event in history_of(entry):
            when = _parse_date(event.get("at"))
            if when is not None and (last is None or when > last):
                last = when
        if last is None:
            continue
        days_since = (today - last).days
        if days_since > stale_days:
            stale.append({
                "id": entry.get("id"),
                "company": entry.get("company"),
                "title": entry.get("title"),
                "status": entry.get("status"),
                "days_since_last_move": days_since,
            })
    stale.sort(key=lambda item: -item["days_since_last_move"])
    return stale


def _best_stage(entry):
    reached = reached_stages(entry)
    best = None
    for stage in FUNNEL_STAGES:
        if stage in reached:
            best = stage
    return best or entry.get("status")


def by_company(entries):
    """Per-company roll-up, most applications first."""
    grouped = {}
    for entry in entries:
        company = entry.get("company") or "(unknown company)"
        row = grouped.setdefault(company, {
            "company": company, "applications": 0, "hired": 0,
            "rejected": 0, "withdrawn": 0, "active": 0, "best_stage": None,
        })
        row["applications"] += 1
        status = entry.get("status")
        if status in ("hired", "rejected", "withdrawn"):
            row[status] += 1
        else:
            row["active"] += 1
        best = _best_stage(entry)
        rank = FUNNEL_STAGES.index(best) if best in FUNNEL_STAGES else -1
        current_rank = (FUNNEL_STAGES.index(row["best_stage"])
                        if row["best_stage"] in FUNNEL_STAGES else -1)
        if rank > current_rank:
            row["best_stage"] = best
    rows = sorted(grouped.values(), key=lambda r: (-r["applications"], r["company"].lower()))
    return rows


def score_stats(entries):
    """Match-score summary, split by outcome where the log allows it."""
    scores = [e["match_score"] for e in entries
              if isinstance(e.get("match_score"), (int, float))]
    if not scores:
        return None
    advanced = [e["match_score"] for e in entries
                if isinstance(e.get("match_score"), (int, float))
                and reached_stages(e) & {"interview", "offer", "hired"}]
    rejected_early = [e["match_score"] for e in entries
                      if isinstance(e.get("match_score"), (int, float))
                      and e.get("status") == "rejected"
                      and reached_stages(e) == {"applied"}]
    stats = {
        "n": len(scores),
        "average": sum(scores) / len(scores),
        "median": _median(scores),
        "min": min(scores),
        "max": max(scores),
    }
    if advanced:
        stats["average_reached_interview"] = sum(advanced) / len(advanced)
        stats["n_reached_interview"] = len(advanced)
    if rejected_early:
        stats["average_rejected_at_applied"] = sum(rejected_early) / len(rejected_early)
        stats["n_rejected_at_applied"] = len(rejected_early)
    return stats


def compute_analytics(entries, today=None, stale_days=14):
    """Full analytics dict for the log (JSON-serialisable)."""
    if today is None:
        today = datetime.date.today()
    data = funnel(entries)
    data["median_days_in_stage"] = days_in_stage(entries)
    data["stale_days"] = stale_days
    data["stale"] = stale_applications(entries, stale_days, today)
    data["by_company"] = by_company(entries)
    data["scores"] = score_stats(entries)
    data["as_of"] = today.isoformat()
    return data


def analytics_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def analytics_report(data):
    """Human-readable funnel report (text)."""
    if data["total"] == 0:
        return ("Applications: 0\n"
                "No applications logged yet — apply to a posting first\n"
                "(see `autojob apply --help`).\n")
    lines = [f"Applications: {data['total']}  (as of {data['as_of']})", "",
             "Funnel — applications that reached each stage:"]
    for stage in FUNNEL_STAGES:
        count = data["reached"][stage]
        rate = data["conversions"][stage]
        if rate is None:
            lines.append(f"  {stage}: {count}")
        else:
            lines.append(f"  {stage}: {count}  ({rate:.1%} of previous stage)")
    lines.append("")
    lines.append(f"Response rate (reached screening): {_pct(data['response_rate'])}")
    lines.append(f"Rejection rate: {_pct(data['rejection_rate'])}")
    lines.append(f"Still active: {data['active']}")
    for status in ("hired", "rejected", "withdrawn"):
        if data["outcomes"].get(status):
            lines.append(f"  {status}: {data['outcomes'][status]}")
    lines.append("")
    if data["median_days_in_stage"]:
        lines.append("Median days in stage (completed stays only):")
        for stage, info in data["median_days_in_stage"].items():
            lines.append(f"  {stage}: {info['median_days']:.1f} days (n={info['n']})")
    else:
        lines.append("Median days in stage: no completed stays recorded yet")
    lines.append("")
    scores = data["scores"]
    if scores:
        lines.append(f"Match scores: average {scores['average']:.2f}, "
                     f"median {scores['median']:.2f}, range "
                     f"{scores['min']:.2f}–{scores['max']:.2f} (n={scores['n']})")
        if "average_reached_interview" in scores:
            lines.append(f"  reached interview+: {scores['average_reached_interview']:.2f} "
                         f"(n={scores['n_reached_interview']})")
        if "average_rejected_at_applied" in scores:
            lines.append(f"  rejected at applied: {scores['average_rejected_at_applied']:.2f} "
                         f"(n={scores['n_rejected_at_applied']})")
        lines.append("")
    if data["stale"]:
        lines.append(f"Stale — no movement in {data['stale_days']}+ days: {len(data['stale'])}")
        for item in data["stale"]:
            lines.append(f"  {item['id']}  {item['company']} — {item['title']}  "
                         f"[{item['status']}]  {item['days_since_last_move']} days")
    else:
        lines.append(f"Stale — no movement in {data['stale_days']}+ days: none")
    lines.append("")
    lines.append("By company:")
    for row in data["by_company"]:
        lines.append(f"  {row['company']}: {row['applications']} application(s), "
                     f"best stage {row['best_stage']}, active {row['active']}, "
                     f"hired {row['hired']}, rejected {row['rejected']}")
    return "\n".join(lines) + "\n"


def _pct(value):
    return f"{value:.1%}" if value is not None else "n/a"
