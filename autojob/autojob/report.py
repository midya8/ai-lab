"""Text and JSON reports for matches and the application log."""

from __future__ import annotations

import json


def match_report(posting, match):
    """Human-readable match report (text)."""
    lines = []
    title = posting.get("title") or "(untitled role)"
    company = posting.get("company") or "(unknown company)"
    lines.append(f"Match: {title} at {company}")
    if match["score"] is None:
        lines.append("Score: n/a — the posting lists no detectable skills")
    else:
        lines.append(f"Score: {match['score']:.2f}  ({match['verdict']})")
    lines.append(f"Required coverage: {_pct(match['required_coverage'])} "
                 f"({len(match['required_matched'])}/{len(match['required_matched']) + len(match['required_missing'])})")
    lines.append(f"Preferred coverage: {_pct(match['preferred_coverage'])}")
    lines.append(f"Experience fit: {_pct(match['experience_fit'])}")
    if match["required_missing"]:
        lines.append("Missing required skills: " + ", ".join(match["required_missing"]))
    if match["preferred_missing"]:
        lines.append("Missing preferred skills: " + ", ".join(match["preferred_missing"]))
    salary = posting.get("salary")
    if salary:
        lines.append(f"Posting salary: {salary['raw']}")
    if posting.get("remote"):
        lines.append(f"Work mode: {posting['remote']}")
    return "\n".join(lines) + "\n"


def match_json(posting, match):
    return json.dumps({"posting": posting, "match": match}, indent=2, sort_keys=True) + "\n"


def log_report(entries):
    """Summary of the application log: counts by status, recent entries."""
    by_status = {}
    for entry in entries:
        by_status[entry["status"]] = by_status.get(entry["status"], 0) + 1
    scored = [e["match_score"] for e in entries if e.get("match_score") is not None]
    lines = [f"Applications: {len(entries)}"]
    for status in ("applied", "screening", "interview", "offer", "hired",
                   "rejected", "withdrawn"):
        if by_status.get(status):
            lines.append(f"  {status}: {by_status[status]}")
    if scored:
        lines.append(f"Average match score of applied roles: {sum(scored) / len(scored):.2f}")
    lines.append("")
    for entry in entries[-10:]:
        lines.append(
            f"{entry['id']}  {entry['applied_at']}  {entry['company']} — {entry['title']}  [{entry['status']}]"
        )
    return "\n".join(lines) + "\n"


def _pct(value):
    return f"{value:.0%}" if value is not None else "n/a"
