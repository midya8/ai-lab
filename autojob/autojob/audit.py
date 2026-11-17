"""Log audit: one read-only pass over everything the log is missing.

``autojob audit`` does not tell you what to do next — that is ``next``.
It tells you what the log itself is missing, stale, or inconsistent
about, so the other commands are working from complete facts: an
application with no source recorded, a past interview round with no
write-up, an offer whose decision deadline has passed while the
application is still live, a rejected application with no reason on
file, a hired application with no onboarding tasks on it. Each issue names the command that fixes it, so the report is a
checklist, not a lecture.

The honest rules, carried over from the modules the checks read:

- A check reads only what is already recorded; it never guesses a
  missing fact. An application with no closing date is listed as
  missing one, not assigned an invented date.
- Dates that cannot be parsed are skipped, never fatal — the same rule
  as every other command. An entry that is not a dict is skipped too.
- The stale bar matches ``analytics.stale_applications`` and the
  follow-up wait: an application whose last activity was exactly
  ``stale_days`` ago is not stale yet. Last activity is the latest
  history event, or ``applied_at`` when there is no history.
- A round dated today already counts as past for the missing-debrief
  check, the same rule as ``debrief.collect_debriefs``: the round
  happened, the write-up is due.
- Terminal applications are left alone, except a rejected one with no
  reason recorded — that is the one thing still worth fixing on a
  closed application, and ``rejections --record`` exists for it. A
  withdrawn application has no rejection reason to record, so it is
  not flagged. The other exception is a hired application with no
  onboarding tasks recorded: the search is over, but the first weeks
  at the new job still need tracking, so the audit notes it and
  names the command that starts the list. A hired application with
  tasks on file is clean.
- An offer deadline is only a problem while the application is still
  live. A hired application decided; a rejected one was decided for
  you. Either way the deadline has nothing left to decide.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES
from .followup import last_activity
from .offer import latest_offer
from .resume import recorded_resume
from .sources import _recorded_source

# The checks, in the order they are defined (the report sorts by
# severity first, then company, then kind).
AUDIT_KINDS = (
    "missing-source",
    "missing-resume",
    "missing-deadline",
    "stale",
    "missing-debrief",
    "missing-rejection-reason",
    "offer-deadline-passed",
    "no-contacts",
    "unmatched-debrief",
    "closing-overdue",
    "missing-onboarding",
)

SEVERITIES = ("error", "warn", "info")
_SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}

# Statuses past applying, where having nobody recorded is worth noting.
_CONTACT_STAGES = ("screening", "interview", "offer")


class AuditError(ValueError):
    """A bad --kind, --severity, or --stale-days for the audit."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _interview_keys(entry):
    """{(date, round): interview} for one entry, first record wins."""
    by_key = {}
    for interview in entry.get("interviews") or []:
        if not isinstance(interview, dict):
            continue
        key = (interview.get("date"), interview.get("round"))
        if key not in by_key:
            by_key[key] = interview
    return by_key


def _debrief_keys(entry):
    """{(date, round)} for every debrief on one entry."""
    keys = set()
    for debrief in entry.get("debriefs") or []:
        if not isinstance(debrief, dict):
            continue
        keys.add((debrief.get("date"), debrief.get("round")))
    return keys


def collect_audit(entries, today=None, stale_days=14, kinds=None,
                  severity=None, app_id=None):
    """Audit every entry and return {"issues": [...], "counts": {...}}.

    ``today`` is a ``datetime.date`` (defaults to the real current
    date; pass an explicit date in tests). ``kinds`` narrows the
    checks to the named kinds (repeatable on the CLI); ``severity``
    keeps only that level; ``app_id`` audits one application only.
    Unknown kinds or severities, a negative ``stale_days``, or an
    unknown ``app_id`` raise AuditError. ``counts`` carries how many
    applications were checked, how many issues there are in total and
    per severity, and per kind.
    """
    if today is None:
        today = datetime.date.today()
    if stale_days is None or stale_days < 0:
        raise AuditError(f"--stale-days must be 0 or more, got {stale_days}")
    if kinds is not None:
        bad = [kind for kind in kinds if kind not in AUDIT_KINDS]
        if bad:
            raise AuditError(f"unknown --kind {bad[0]!r}; one of: "
                             f"{', '.join(AUDIT_KINDS)}")
        wanted = set(kinds)
    else:
        wanted = set(AUDIT_KINDS)
    if severity is not None and severity not in SEVERITIES:
        raise AuditError(f"--severity must be one of "
                         f"{', '.join(SEVERITIES)}, got {severity!r}")

    issues = []
    checked = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if app_id is not None and entry.get("id") != app_id:
            continue
        checked += 1
        status = entry.get("status")
        live = status not in TERMINAL_STATUSES
        app_id_value = entry.get("id")
        company = entry.get("company")
        title = entry.get("title")

        def add(kind, level, detail, days=None):
            if kind not in wanted:
                return
            if severity is not None and level != severity:
                return
            issues.append({
                "severity": level,
                "kind": kind,
                "id": app_id_value,
                "company": company,
                "title": title,
                "status": status,
                "detail": detail,
                "days": days,
            })

        if live:
            if _recorded_source(entry) is None:
                add("missing-source", "warn",
                    "no source recorded — set one with "
                    f"`sources --record --id {app_id_value} "
                    "--source job-board`")
            if recorded_resume(entry) is None:
                add("missing-resume", "warn",
                    "no resume recorded — set one with "
                    f"`resume --record --id {app_id_value} "
                    "--file resume.pdf`")
            if not entry.get("deadline"):
                add("missing-deadline", "info",
                    "no closing date recorded — set one with "
                    f"`deadlines --record --id {app_id_value} "
                    "--date YYYY-MM-DD`")
            last = last_activity(entry)
            if last is not None:
                days_quiet = (today - last).days
                if days_quiet > stale_days:
                    add("stale", "warn",
                        f"no activity for {days_quiet} days "
                        f"(last: {last.isoformat()}, bar {stale_days}) — "
                        "consider a follow-up or a status move",
                        days=days_quiet)
            deadline = _parse_date(entry.get("deadline"))
            if deadline is not None and deadline < today:
                days_over = (today - deadline).days
                add("closing-overdue", "warn",
                    f"posting closed {days_over} day"
                    f"{'s' if days_over != 1 else ''} ago "
                    f"({deadline.isoformat()}) and the application is "
                    "still live",
                    days=days_over)
            if status in _CONTACT_STAGES and not entry.get("contacts"):
                add("no-contacts", "info",
                    "no one recorded for this application — add a person "
                    f"with `contacts --record --id {app_id_value} "
                    "--name NAME --date YYYY-MM-DD`")
            offer = latest_offer(entry)
            if offer is not None:
                offer_deadline = _parse_date(offer.get("deadline"))
                if (offer_deadline is not None
                        and offer_deadline < today):
                    days_over = (today - offer_deadline).days
                    add("offer-deadline-passed", "error",
                        f"offer decision deadline passed {days_over} day"
                        f"{'s' if days_over != 1 else ''} ago "
                        f"({offer_deadline.isoformat()}) and the "
                        "application is still live — move it with "
                        f"`status --id {app_id_value} --to ...`",
                        days=days_over)
            interviews = _interview_keys(entry)
            debriefed = _debrief_keys(entry)
            for (when_str, round_label), interview in interviews.items():
                when = _parse_date(when_str)
                if when is None or when > today:
                    continue
                if (when_str, round_label) in debriefed:
                    continue
                days_ago = (today - when).days
                add("missing-debrief", "warn",
                    f"round {round_label!r} on {when_str} "
                    f"({days_ago} day{'s' if days_ago != 1 else ''} ago) "
                    "has no write-up — record one with "
                    f"`debrief --record --id {app_id_value} "
                    f"--date {when_str} --round {round_label!r}`",
                    days=days_ago)
            for debrief in entry.get("debriefs") or []:
                if not isinstance(debrief, dict):
                    continue
                key = (debrief.get("date"), debrief.get("round"))
                if key in interviews:
                    continue
                add("unmatched-debrief", "info",
                    f"debrief for round {debrief.get('round')!r} on "
                    f"{debrief.get('date')} has no matching interview "
                    "round recorded — record the round, or check the "
                    "date and label")
        if status == "rejected" and not entry.get("rejection"):
            add("missing-rejection-reason", "warn",
                "rejected but no reason recorded — set one with "
                f"`rejections --record --id {app_id_value} "
                "--reason ...`")
        if status == "hired":
            tasks = entry.get("onboarding")
            if not isinstance(tasks, list) or not tasks:
                add("missing-onboarding", "info",
                    "hired but no onboarding tasks recorded — start "
                    "the list with "
                    f"`onboard --record --id {app_id_value} "
                    "--task 'Complete I-9 paperwork' "
                    "--due YYYY-MM-DD`")

    def order(issue):
        return (_SEVERITY_RANK[issue["severity"]],
                (issue["company"] or "").lower(),
                (issue["title"] or "").lower(),
                issue["kind"],
                issue["detail"])

    issues.sort(key=order)
    by_kind = {}
    for issue in issues:
        by_kind[issue["kind"]] = by_kind.get(issue["kind"], 0) + 1
    counts = {
        "checked": checked,
        "issues": len(issues),
        "error": sum(1 for i in issues if i["severity"] == "error"),
        "warn": sum(1 for i in issues if i["severity"] == "warn"),
        "info": sum(1 for i in issues if i["severity"] == "info"),
        "by_kind": by_kind,
    }
    return {"issues": issues, "counts": counts}


def audit_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def render_audit_markdown(data):
    """Human-readable audit: errors first, then warnings, then notes."""
    counts = data["counts"]
    issues = data["issues"]
    lines = [f"Audit — {counts['checked']} application(s) checked, "
             f"{counts['issues']} issue(s) "
             f"({counts['error']} error(s), {counts['warn']} warning(s), "
             f"{counts['info']} note(s))"]
    if not issues:
        lines.append("Nothing to fix — every live application has a source, "
                     "a resume, and a closing date recorded, nothing is "
                     "stale, and every past round has a write-up.")
        return "\n".join(lines) + "\n"
    lines.append("An error is a decision that is overdue; a warning is "
                 "something missing or gone quiet; a note is worth knowing "
                 "but not urgent. Each line names the command that fixes it.")
    lines.append("")
    for issue in issues:
        lines.append(
            f"  [{issue['severity']}] {issue['id']}  "
            f"{issue['company']} — {issue['title']}  [{issue['status']}]  "
            f"{issue['kind']}: {issue['detail']}")
    if counts["by_kind"]:
        by_kind = ", ".join(
            f"{kind} {n}" for kind, n in sorted(counts["by_kind"].items()))
        lines.append("")
        lines.append(f"By kind: {by_kind}")
    return "\n".join(lines) + "\n"
