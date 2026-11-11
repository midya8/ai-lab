"""Weekly digest: one page over the whole application log.

A digest combines the funnel (see analytics), the follow-ups due (see
followup), and a gap leaderboard — the skills postings keep asking for
that the profile does not have, counted across every application that
recorded them. It is the view a job seeker actually wants once a week:
what is moving, what has gone quiet, and what to learn next.

Skill data comes only from tracker entries written by ``apply`` since
v0.6, which records the posting's required/preferred skills and which of
them the profile matched. Entries logged before that carry no skill
lists; the digest says how many it skipped instead of guessing — a
leaderboard built on half the log and labelled as if it covered all of
it would be dishonest.
"""

from __future__ import annotations

import datetime
import json

from .analytics import compute_analytics
from .followup import due_followups

SKILL_FIELDS = ("required_matched", "required_missing",
                "preferred_matched", "preferred_missing")


def has_skill_data(entry):
    """True when an entry recorded posting skills at apply time (v0.6+)."""
    return any(isinstance(entry.get(field), list) for field in SKILL_FIELDS)


def gap_leaderboard(entries):
    """Skills postings ask for that the profile lacks, most-missed first.

    Each row counts, over entries with skill data, how many applications
    asked for the skill (required and preferred separately) and how many
    of those it was a gap in, plus the companies that asked for it while
    it was missing. Rows sort by total missing first, then total asked,
    then skill name — so the top row is the gap costing the most
    applications, not the alphabetically first one. Entries without skill
    data are skipped; see ``build_digest`` for the count.
    """
    rows = {}
    for entry in entries:
        if not has_skill_data(entry):
            continue
        company = entry.get("company") or "(unknown company)"
        for kind, matched_field, missing_field in (
                ("required", "required_matched", "required_missing"),
                ("preferred", "preferred_matched", "preferred_missing")):
            matched = entry.get(matched_field) or []
            missing = entry.get(missing_field) or []
            for skill in list(matched) + list(missing):
                row = rows.setdefault(skill, {
                    "skill": skill,
                    "asked_required": 0, "missing_required": 0,
                    "asked_preferred": 0, "missing_preferred": 0,
                    "companies": [],
                })
                row[f"asked_{kind}"] += 1
                if skill in missing:
                    row[f"missing_{kind}"] += 1
                    if company not in row["companies"]:
                        row["companies"].append(company)
    for row in rows.values():
        row["companies"].sort()
        row["asked"] = row["asked_required"] + row["asked_preferred"]
        row["missing"] = row["missing_required"] + row["missing_preferred"]
    # Only gaps belong on the leaderboard: a skill the profile has in
    # every application that asked for it is not something to learn.
    return sorted((row for row in rows.values() if row["missing"] > 0),
                  key=lambda r: (-r["missing"], -r["asked"], r["skill"].lower()))


def build_digest(entries, today=None, stale_days=14, wait_days=None):
    """Full digest dict for the log (JSON-serialisable)."""
    if today is None:
        today = datetime.date.today()
    analytics = compute_analytics(entries, today=today, stale_days=stale_days)
    with_skills = [e for e in entries if has_skill_data(e)]
    return {
        "generated_on": today.isoformat(),
        "n_applications": len(entries),
        "n_with_skill_data": len(with_skills),
        "n_without_skill_data": len(entries) - len(with_skills),
        "funnel": {
            "total": analytics["total"],
            "reached": analytics["reached"],
            "conversions": analytics["conversions"],
            "outcomes": analytics["outcomes"],
            "active": analytics["active"],
            "response_rate": analytics["response_rate"],
            "rejection_rate": analytics["rejection_rate"],
        },
        "median_days_in_stage": analytics["median_days_in_stage"],
        "scores": analytics["scores"],
        "stale_days": stale_days,
        "stale": analytics["stale"],
        "due_followups": due_followups(entries, today, wait_days=wait_days),
        "gap_leaderboard": gap_leaderboard(entries),
    }


def digest_json(digest):
    return json.dumps(digest, indent=2, sort_keys=True) + "\n"


def _pct(value):
    return f"{value:.1%}" if value is not None else "n/a"


def render_digest_markdown(digest):
    """Human-readable one-page digest (text)."""
    n = digest["n_applications"]
    if n == 0:
        return ("Application digest — no applications logged yet\n"
                "Apply to a posting first (see `autojob apply --help`).\n")
    funnel = digest["funnel"]
    lines = [f"Application digest — {digest['generated_on']}",
             f"Applications: {n}  ({funnel['active']} still active)", "",
             "Funnel — applications that reached each stage:"]
    for stage in ("applied", "screening", "interview", "offer", "hired"):
        count = funnel["reached"][stage]
        rate = funnel["conversions"][stage]
        if rate is None:
            lines.append(f"  {stage}: {count}")
        else:
            lines.append(f"  {stage}: {count}  ({rate:.1%} of previous stage)")
    lines.append("")
    lines.append(f"Response rate (reached screening): {_pct(funnel['response_rate'])}")
    lines.append(f"Rejection rate: {_pct(funnel['rejection_rate'])}")
    lines.append("")
    scores = digest["scores"]
    if scores:
        lines.append(f"Match scores: average {scores['average']:.2f}, "
                     f"median {scores['median']:.2f}, range "
                     f"{scores['min']:.2f}–{scores['max']:.2f} (n={scores['n']})")
        lines.append("")
    if digest["median_days_in_stage"]:
        lines.append("Median days in stage (completed stays only):")
        for stage, info in digest["median_days_in_stage"].items():
            lines.append(f"  {stage}: {info['median_days']:.1f} days (n={info['n']})")
        lines.append("")
    if digest["stale"]:
        lines.append(f"Stale — no movement in {digest['stale_days']}+ days: "
                     f"{len(digest['stale'])}")
        for item in digest["stale"]:
            lines.append(f"  {item['id']}  {item['company']} — {item['title']}  "
                         f"[{item['status']}]  {item['days_since_last_move']} days")
    else:
        lines.append(f"Stale — no movement in {digest['stale_days']}+ days: none")
    lines.append("")
    due = digest["due_followups"]
    if due:
        lines.append(f"Due a follow-up: {len(due)}")
        for item in due:
            lines.append(f"  {item['id']}  {item['company']} — {item['title']}  "
                         f"[{item['status']}]  waiting {item['days_waiting']} days")
        lines.append("  (Drafts: `autojob followup --profile <profile.json>`)")
    else:
        lines.append("Due a follow-up: none")
    lines.append("")
    board = digest["gap_leaderboard"]
    if board:
        lines.append("Most-asked-for skills you don't have yet "
                     f"(n={digest['n_with_skill_data']} applications with skill data):")
        for i, row in enumerate(board, 1):
            parts = []
            if row["missing_required"]:
                parts.append(f"required in {row['missing_required']} of "
                             f"{row['asked_required']}")
            if row["missing_preferred"]:
                parts.append(f"preferred in {row['missing_preferred']} of "
                             f"{row['asked_preferred']}")
            lines.append(f"  {i}. {row['skill']} — missing in "
                         + "; ".join(parts)
                         + f"; asked by: {', '.join(row['companies'])}")
    elif digest["n_with_skill_data"]:
        lines.append("Most-asked-for skills you don't have yet: none — "
                     "every skill the postings asked for is in the profile")
    else:
        lines.append("Most-asked-for skills you don't have yet: no skill data yet")
    if digest["n_without_skill_data"]:
        lines.append(f"  (Note: {digest['n_without_skill_data']} application(s) logged "
                     "before v0.6 carry no skill data and are not counted above.)")
    return "\n".join(lines) + "\n"
