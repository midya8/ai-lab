"""Data export: the whole application log as a spreadsheet or JSON.

``autojob export`` gets your data out of the log in one row per
application — company, title, status, dates, match score, the skills
each posting asked for and which ones you had, source and referrer,
closing date, rejection reason, latest offer figures, and counts of
interviews, contacts, and follow-ups sent. The point is a file you
can open in a spreadsheet or hand to another tool, not another view
inside autojob.

The honest rules, carried over from the other modules:

- The export is read-only: it records nothing, changes nothing, and
  never touches the log. It is a report over what is already there.
- A missing value stays missing. A skill list the log never recorded
  is an empty cell in CSV and ``null`` in JSON, not a guess; an
  application with no offer has empty offer columns; an entry with
  no parseable ``applied_at`` sorts last and carries no day counts.
- Day counts come from the recorded history only. ``days_since_applied``
  is today minus ``applied_at``; ``days_in_current_stage`` is today
  minus the last recorded status change — a follow-up, interview, or
  contact recorded mid-stage does not restart the stage, the same
  rule analytics uses, so a nudge never makes a stage look shorter
  than it is.
- The latest offer is the one ``offer.latest_offer`` picks (most
  recently recorded), and its first-year total is the same figure
  ``offer.first_year_total`` reports: base plus bonus plus equity
  per vesting year, with equity counted only when a vesting schedule
  is given. Currencies are never converted and nothing is annualised
  here — the base shown is the annualised figure the offer already
  carries.
- Rows are newest applied first. ``--status`` and ``--source`` narrow
  the rows; a filter value that is not a real status or channel is
  refused, not silently matched to nothing.
"""

from __future__ import annotations

import csv
import datetime
import io
import json

from .analytics import history_of
from .followup import followups_sent, last_activity
from .offer import first_year_total, latest_offer
from .sources import SOURCE_CHOICES
from .tracker import STATUSES

EXPORT_FORMATS = ("csv", "json")

COLUMNS = (
    "id",
    "company",
    "title",
    "status",
    "applied_at",
    "days_since_applied",
    "days_in_current_stage",
    "match_score",
    "required_matched",
    "required_missing",
    "preferred_matched",
    "preferred_missing",
    "source",
    "referrer",
    "deadline",
    "rejection_reason",
    "latest_offer_base",
    "latest_offer_currency",
    "latest_offer_first_year_total",
    "interviews",
    "contacts",
    "followups_sent",
    "last_activity_at",
)

LIST_COLUMNS = (
    "required_matched",
    "required_missing",
    "preferred_matched",
    "preferred_missing",
)


class ExportError(ValueError):
    """An export that cannot be built as asked."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _count(items):
    return len(items) if isinstance(items, list) else 0


def _days_in_current_stage(entry, today):
    """Days since the last recorded status change (or applied_at)."""
    history = [h for h in history_of(entry) if h.get("status")]
    start = None
    for event in history:
        when = _parse_date(event.get("at"))
        if when is not None and (start is None or when > start):
            start = when
    if start is None:
        start = _parse_date(entry.get("applied_at"))
    if start is None:
        return None
    return (today - start).days


def _row(entry, today):
    applied = _parse_date(entry.get("applied_at"))
    offer = latest_offer(entry)
    if offer is not None:
        first_year, _counted = first_year_total(offer)
        base = offer.get("base_annual")
        currency = offer.get("currency")
    else:
        first_year = None
        base = None
        currency = None
    rejection = entry.get("rejection")
    reason = (rejection.get("category")
              if isinstance(rejection, dict) else None)
    last = last_activity(entry)
    row = {
        "id": entry.get("id"),
        "company": entry.get("company"),
        "title": entry.get("title"),
        "status": entry.get("status"),
        "applied_at": entry.get("applied_at"),
        "days_since_applied": (
            (today - applied).days if applied is not None else None),
        "days_in_current_stage": _days_in_current_stage(entry, today),
        "match_score": entry.get("match_score"),
        "required_matched": entry.get("required_matched"),
        "required_missing": entry.get("required_missing"),
        "preferred_matched": entry.get("preferred_matched"),
        "preferred_missing": entry.get("preferred_missing"),
        "source": entry.get("source"),
        "referrer": entry.get("referrer"),
        "deadline": entry.get("deadline"),
        "rejection_reason": reason,
        "latest_offer_base": base,
        "latest_offer_currency": currency,
        "latest_offer_first_year_total": first_year,
        "interviews": _count(entry.get("interviews")),
        "contacts": _count(entry.get("contacts")),
        "followups_sent": followups_sent(entry),
        "last_activity_at": last.isoformat() if last is not None else None,
    }
    return row, applied


def collect_export(entries, today=None, status=None, source=None):
    """One row per application, newest applied first.

    Returns {"rows": [...], "count": n, "today": iso}. Each row
    carries every column in COLUMNS, in that order; skill lists stay
    lists (CSV joins them with "; ", JSON keeps them). ``today`` is
    a ``datetime.date`` (defaults to the real current date; pass an
    explicit date in tests). ``status`` and ``source`` narrow the
    rows to one status or channel. A non-dict entry is skipped,
    never fatal; a bad filter value raises ExportError.
    """
    if today is None:
        today = datetime.date.today()
    if status is not None and status not in STATUSES:
        raise ExportError(
            f"status must be one of {', '.join(STATUSES)}, got {status!r}")
    if source is not None and source not in SOURCE_CHOICES:
        raise ExportError(
            f"source must be one of {', '.join(SOURCE_CHOICES)}, "
            f"got {source!r}")
    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if status is not None and entry.get("status") != status:
            continue
        if source is not None and entry.get("source") != source:
            continue
        row, applied = _row(entry, today)
        rows.append((row, applied))
    # Newest applied first; entries with no parseable date sort last.
    # Stable sort keeps the log order (then id order) within one date.
    rows.sort(key=lambda pair: pair[0].get("id") or "")
    rows.sort(key=lambda pair: (pair[1] is not None,
                                pair[1] or datetime.date.min),
              reverse=True)
    return {"rows": [row for row, _ in rows],
            "count": len(rows), "today": today.isoformat()}


def _csv_value(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def export_csv(data):
    """The export as CSV text: header row plus one row per application."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in data["rows"]:
        writer.writerow([_csv_value(row.get(column)) for column in COLUMNS])
    return buf.getvalue()


def export_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
