"""Posting closing dates: which application to finish first.

``autojob deadlines`` lists the closing date of every live application,
soonest first, so the posting that stops taking applications tomorrow is
the one you finish tonight. A closing date is recorded at apply time
(``apply --deadline YYYY-MM-DD``) or later (``deadlines --record --id
APP-001 --date YYYY-MM-DD``).

The honest rules, carried over from the other modules:

- A closing date is a fact you supply, about the posting — not an event
  that happened to the application. It is stored as a field on the
  tracker entry (see ``tracker.record_deadline``), not as a history
  event: recording one does not reset the follow-up or stale clocks,
  and it does not appear in the timeline feed. The follow-up clock
  measures contact with the employer; looking up a closing date is
  not contact.
- A past date is allowed, not refused: a posting can already be closed
  when you record it, and the list flags it overdue instead of hiding
  it. Overdue means the closing date passed while the application is
  still live — the posting may have closed under you.
- "Closing soon" is 1–7 days out; today is its own state, and anything
  further out is "later". Exactly 7 days out is soon, not later.
  ``--days N`` narrows the list to deadlines N days out or sooner;
  overdue ones are always shown, because they are the urgent ones.
- Applications with no closing date recorded are counted, not guessed:
  the list says how many live applications carry no date, instead of
  inventing one.
- A closing date on a terminal application (hired / rejected /
  withdrawn) is left out of the list: there is nothing left to submit
  for it. Recording one is refused for the same reason.
- A closing date that will not parse — a hand-edited log line — is
  skipped, never fatal.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES

SOON_DAYS = 7


class DeadlineError(ValueError):
    """A closing date that cannot be recorded as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def validate_deadline(date):
    """Validate a closing date and return it unchanged.

    The date is required and must be YYYY-MM-DD. A date in the past is
    allowed: a posting can already be closed when you record it, and
    the list flags it overdue rather than refusing a fact.
    """
    if date is None or (isinstance(date, str) and not date.strip()):
        raise DeadlineError("closing date is required (YYYY-MM-DD)")
    if not isinstance(date, str):
        raise DeadlineError(f"closing date must be YYYY-MM-DD, got {date!r}")
    if _parse_date(date) is None:
        raise DeadlineError(f"closing date must be YYYY-MM-DD, got {date!r}")
    return date


def _state(days_left):
    if days_left < 0:
        return "overdue"
    if days_left == 0:
        return "today"
    if days_left <= SOON_DAYS:
        return "soon"
    return "later"


def collect_deadlines(entries, today=None, days=None, overdue_only=False):
    """Live applications with a closing date, soonest first.

    Returns {"deadlines": [...], "counts": {...}}. Each row carries the
    entry's identity, its stage, the closing date, days left (negative
    when overdue), and its state. ``today`` is a ``datetime.date``
    (defaults to the real current date; pass an explicit date in
    tests). ``days`` narrows the list to deadlines that many days out
    or sooner — overdue rows always pass, since their days-left is
    negative. ``overdue_only`` keeps only the overdue rows. Entries
    without a usable closing date are counted under
    ``without_deadline`` instead of being guessed at.
    """
    if today is None:
        today = datetime.date.today()
    if days is not None and days < 0:
        raise ValueError(f"--days must be 0 or more, got {days}")
    rows = []
    without_deadline = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        deadline = _parse_date(entry.get("deadline"))
        if deadline is None:
            without_deadline += 1
            continue
        days_left = (deadline - today).days
        if overdue_only and days_left >= 0:
            continue
        if days is not None and days_left > days:
            continue
        rows.append({
            "id": entry.get("id"),
            "company": entry.get("company"),
            "title": entry.get("title"),
            "status": entry.get("status"),
            "deadline": deadline.isoformat(),
            "days_left": days_left,
            "state": _state(days_left),
        })
    rows.sort(key=lambda row: (row["deadline"],
                               (row["company"] or "").lower(),
                               (row["title"] or "").lower()))
    return {
        "deadlines": rows,
        "counts": {
            "total": len(rows),
            "overdue": sum(1 for row in rows if row["state"] == "overdue"),
            "today": sum(1 for row in rows if row["state"] == "today"),
            "soon": sum(1 for row in rows if row["state"] == "soon"),
            "later": sum(1 for row in rows if row["state"] == "later"),
            "without_deadline": without_deadline,
        },
    }


def deadlines_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    if row["days_left"] < 0:
        when = f"closed {-row['days_left']}d ago"
    elif row["days_left"] == 0:
        when = "closes today"
    elif row["days_left"] == 1:
        when = "closes tomorrow"
    else:
        when = f"closes in {row['days_left']}d"
    line = (f"{row['id']}  {row['company']} — {row['title']}  "
            f"[{row['status']}]  {row['deadline']} — {when}")
    if row["state"] == "overdue":
        line += "  [overdue]"
    return line


def render_deadlines_markdown(data):
    """Human-readable closing-dates list: soonest first."""
    rows = data["deadlines"]
    counts = data["counts"]
    header = (f"Closing dates — {counts['total']} application"
              + ("s" if counts["total"] != 1 else "")
              + " with a date recorded")
    if counts["overdue"]:
        header += (f", {counts['overdue']} overdue")
    if counts["today"]:
        header += (f", {counts['today']} closing today")
    if counts["soon"]:
        header += (f", {counts['soon']} closing soon")
    if counts["without_deadline"]:
        header += (f"; {counts['without_deadline']} live application"
                   + ("s" if counts["without_deadline"] != 1 else "")
                   + " with no closing date recorded")
    lines = [header]
    if not rows:
        lines.append("No closing dates recorded yet — use "
                     "`apply --deadline YYYY-MM-DD` when you apply, or "
                     "`deadlines --record --id APP-001 --date YYYY-MM-DD` "
                     "to add one later.")
        return "\n".join(lines) + "\n"
    lines.append("")
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
