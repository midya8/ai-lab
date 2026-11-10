"""Follow-up reminders for applications that have gone quiet.

An application is "due a follow-up" when its current stage has had no
recorded activity for longer than the wait for that stage. Activity means
any recorded event: a status move or a follow-up already sent. Terminal
applications (hired / rejected / withdrawn) are never due.

The waits are judgement calls, documented as such — a week after applying
or reaching screening, three days after an interview, two days at offer
stage — and ``--wait-days`` overrides them all with a single number. The
boundary matches ``analytics.stale_applications``: an application whose
last activity was exactly ``wait`` days ago is not due yet.

The draft follow-up email uses only facts already in the tracker entry
(company, title, applied date) plus the applicant's name. It invents no
recruiter name — the greeting is "Hiring team" — and claims nothing about
the application's progress, because the tracker does not know it.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES, history_of

# Default wait (days) before a follow-up is due, per current stage.
STAGE_WAIT_DAYS = {
    "applied": 7,
    "screening": 7,
    "interview": 3,
    "offer": 2,
}


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def wait_for(status, wait_days=None):
    """Wait in days before a follow-up is due at ``status``.

    ``wait_days`` overrides the per-stage default for every stage. A
    negative wait makes no sense and is refused.
    """
    if wait_days is not None:
        if wait_days < 0:
            raise ValueError(f"--wait-days must be 0 or more, got {wait_days}")
        return wait_days
    return STAGE_WAIT_DAYS.get(status, 7)


def last_activity(entry):
    """Date of the latest recorded event, or ``applied_at`` if none."""
    last = None
    for event in history_of(entry):
        when = _parse_date(event.get("at"))
        if when is not None and (last is None or when > last):
            last = when
    if last is None:
        last = _parse_date(entry.get("applied_at"))
    return last


def followups_sent(entry):
    """How many follow-ups have already been recorded for this entry."""
    return sum(1 for event in history_of(entry)
               if event.get("kind") == "followup")


def due_followups(entries, today, wait_days=None):
    """Applications due a follow-up, most overdue first.

    Each item carries the facts the CLI and the draft need: the entry's
    identity, its stage, how long it has waited, and how many follow-ups
    have already been sent (so a second nudge is never mistaken for the
    first). ``today`` is a ``datetime.date``.
    """
    due = []
    for entry in entries:
        status = entry.get("status")
        if status in TERMINAL_STATUSES:
            continue
        last = last_activity(entry)
        if last is None:
            continue
        wait = wait_for(status, wait_days)
        days_waiting = (today - last).days
        if days_waiting <= wait:
            continue
        sent = followups_sent(entry)
        due.append({
            "id": entry.get("id"),
            "company": entry.get("company"),
            "title": entry.get("title"),
            "status": status,
            "applied_at": entry.get("applied_at"),
            "last_activity": last.isoformat(),
            "days_waiting": days_waiting,
            "wait_days": wait,
            "followups_sent": sent,
            "suggestion": ("follow up again" if sent else "send a first follow-up"),
        })
    due.sort(key=lambda item: -item["days_waiting"])
    return due


def draft_followup(entry, name):
    """A short follow-up email built only from tracker facts + ``name``.

    Returns {"subject": ..., "body": ...}. The body names the role, the
    company, and the date applied — nothing else about the application,
    because nothing else is known.
    """
    company = entry.get("company") or "unknown company"
    title = entry.get("title") or "unknown role"
    applied = entry.get("applied_at") or "an earlier date"
    subject = f"Following up — {title} application ({company})"
    body = (
        "Hello Hiring team,\n"
        "\n"
        f"I'm {name}. I applied for the {title} role at {company} on "
        f"{applied}, and wanted to check in on the status of my application.\n"
        "\n"
        "I'm still very interested in the role and happy to provide anything "
        "else that would help your review.\n"
        "\n"
        "Thank you for your time,\n"
        f"{name}\n"
    )
    return {"subject": subject, "body": body}


def followup_json(items, name):
    """JSON rendering of due follow-ups, each with its draft email.

    ``name`` is the applicant's name from their profile, or None — then
    each item's ``draft`` is null instead of an invented greeting.
    """
    enriched = []
    for item in items:
        row = dict(item)
        row["draft"] = (draft_followup(
            {"company": item["company"], "title": item["title"],
             "applied_at": item["applied_at"]},
            name) if name else None)
        enriched.append(row)
    return json.dumps(enriched, indent=2, sort_keys=True) + "\n"


def followup_report(items, name):
    """Human-readable list of due follow-ups, each with its draft email.

    With no ``name`` (no --profile passed), drafts are omitted and the
    report says so, rather than drafting a letter to nobody.
    """
    if not items:
        return ("Follow-ups due: none\n"
                "No application has waited past its follow-up window.\n")
    lines = [f"Follow-ups due: {len(items)}", ""]
    for item in items:
        lines.append(
            f"{item['id']}  {item['company']} — {item['title']}  "
            f"[{item['status']}]  waiting {item['days_waiting']} days "
            f"(wait {item['wait_days']}) — {item['suggestion']}"
            + (f" ({item['followups_sent']} sent already)"
               if item["followups_sent"] else ""))
        if name:
            draft = draft_followup(
                {"company": item["company"], "title": item["title"],
                 "applied_at": item["applied_at"]},
                name)
            lines.append(f"  Subject: {draft['subject']}")
            for body_line in draft["body"].splitlines():
                lines.append(f"  {body_line}" if body_line else "")
        lines.append("")
    if not name:
        lines.append("(Pass --profile <profile.json> to include a draft "
                     "follow-up email for each application.)")
    return "\n".join(lines)
