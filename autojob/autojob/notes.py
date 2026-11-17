"""Application notes: what you learn along the way, written down dated.

``autojob notes`` records a dated note on an application — something you
learned about the company, a detail from a call, a reminder to yourself
before the next round. The note is your own words, stored verbatim.

The honest rules, carried over from the other modules:

- A note is a fact you supply: the date it belongs to and the text.
  The date is required and never defaulted (the same rule as
  contacts): a note dated today by default would be wrong whenever
  you write up something you learned yesterday. A date in the future
  is refused, the same rule as debriefs — a note is written about
  something already learned, not something still to happen.
- A note is an event, like a contact or a debrief: it is stored as a
  history event (see ``tracker.record_note``), so writing one counts
  as activity — it resets the follow-up and stale clocks — and it
  appears in the timeline feed. It never splits a stage stay in the
  analytics: stays are measured between status changes only.
- Recording a note does not move the application's status — that
  stays explicit through ``status``, the same rule as offers,
  interviews, and contacts. Recording is refused on a terminal
  application (hired / rejected / withdrawn): the application is
  closed, and a rejection reason is the sanctioned annotation for
  a rejected one.
- A tag is an optional free-text label ("company", "interview",
  "salary") so the list can be narrowed; it is stored verbatim apart
  from surrounding whitespace and matched case-insensitively. A note
  with no tag is simply untagged, never given one.
- Notes are searchable like any other recorded fact (see
  ``search.searchable_fields``), and the export counts the notes on
  each application. Notes on terminal applications are left out of
  the list, the same rule as the contacts and debriefs lists.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES


class NoteError(ValueError):
    """A note that cannot be recorded as stated."""


def validate_note(data, today=None):
    """Validate a note dict and return the cleaned form.

    Every problem is reported at once, like profile, offer,
    interview, contact, and debrief validation. ``today`` defaults
    to the real current date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    errors = []
    data = data or {}

    date = data.get("date")
    when = None
    if date is None or (isinstance(date, str) and not date.strip()):
        errors.append("date is required (the date the note belongs to)")
    elif not isinstance(date, str):
        errors.append(f"date must be YYYY-MM-DD, got {date!r}")
    else:
        try:
            when = datetime.date.fromisoformat(date)
        except ValueError:
            errors.append(f"date must be YYYY-MM-DD, got {date!r}")
    if when is not None:
        try:
            ref = datetime.date.fromisoformat(today)
        except (ValueError, TypeError):
            ref = None
        if ref is not None and when > ref:
            errors.append("date cannot be in the future — a note is "
                          "written about something already learned")

    text = data.get("text")
    if text is None or (isinstance(text, str) and not text.strip()):
        errors.append("text is required (the note itself)")
    elif not isinstance(text, str):
        errors.append(f"text must be text, got {text!r}")

    tag = data.get("tag")
    cleaned_tag = None
    if tag is not None:
        if not isinstance(tag, str):
            errors.append(f"tag must be text, got {tag!r}")
        elif tag.strip():
            cleaned_tag = tag.strip()

    if errors:
        raise NoteError("; ".join(errors))

    return {
        "date": date.strip(),
        "text": text.strip(),
        "tag": cleaned_tag,
        "recorded_at": today,
    }


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _clean_note(note):
    """A stored note as a list row, or None when it is not one.

    A note that is not a dict, or carries no parseable date or
    non-empty text (a hand-edited log), is skipped, never fatal.
    """
    if not isinstance(note, dict):
        return None
    when = _parse_date(note.get("date"))
    if when is None:
        return None
    text = note.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    tag = note.get("tag")
    if not isinstance(tag, str) or not tag.strip():
        tag = None
    else:
        tag = tag.strip()
    return {"date": note["date"], "text": text.strip(), "tag": tag,
            "when": when}


def collect_notes(entries, today=None, tag=None, app_id=None):
    """Every recorded note, newest first.

    Returns {"notes": [...], "summary": {...}}. Each row carries the
    application id, company, title, status, date, tag, text, and
    days since the note. Rows are newest first; notes dated the same
    day come in the order they were recorded, latest first. The
    summary counts the notes and rolls them up by tag. ``tag``
    narrows the rows to notes carrying that tag (case-insensitive,
    surrounding whitespace ignored); ``app_id`` shows one
    application. Notes on terminal applications are left out, the
    same rule as the contacts list. ``today`` is a ``datetime.date``
    (defaults to the real current date; pass an explicit date in
    tests).
    """
    if today is None:
        today = datetime.date.today()
    wanted_tag = tag.strip().lower() if isinstance(tag, str) else None
    rows = []
    by_tag = {}
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        for note in entry.get("notes") or []:
            cleaned = _clean_note(note)
            if cleaned is None:
                continue
            total += 1
            if cleaned["tag"] is not None:
                by_tag[cleaned["tag"]] = by_tag.get(cleaned["tag"], 0) + 1
            if app_id is not None and entry.get("id") != app_id:
                continue
            if wanted_tag is not None and (
                    cleaned["tag"] is None
                    or cleaned["tag"].lower() != wanted_tag):
                continue
            rows.append({
                "id": entry.get("id"),
                "company": entry.get("company"),
                "title": entry.get("title"),
                "status": entry.get("status"),
                "date": cleaned["date"],
                "tag": cleaned["tag"],
                "text": cleaned["text"],
                "days_ago": (today - cleaned["when"]).days,
            })
    # Newest first; within a day the latest-recorded note comes
    # first, so the list reads like a diary kept backwards.
    rows.sort(key=lambda row: row["date"], reverse=True)
    rows = _reverse_within_days(rows)
    return {
        "notes": rows,
        "summary": {
            "total": total,
            "by_tag": dict(sorted(by_tag.items(),
                                  key=lambda pair: (-pair[1], pair[0]))),
        },
    }


def _reverse_within_days(rows):
    """Reverse each same-day run, so the latest-recorded note that day
    comes first. ``rows`` arrives sorted by date descending."""
    out = []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["date"] == rows[i]["date"]:
            j += 1
        out.extend(reversed(rows[i:j]))
        i = j
    return out


def notes_for(entries, app_id, today=None):
    """All notes for one application.

    Returns the entry plus its note rows, or (None, []) when the id
    is unknown. The entry is returned even when it holds no notes,
    so the caller can tell "known application, nothing written"
    apart from "no such application". Terminal applications return
    their entry with an empty list, the same rule as the full list.
    """
    if today is None:
        today = datetime.date.today()
    for entry in entries:
        if entry.get("id") == app_id:
            data = collect_notes([entry], today=today)
            return entry, data["notes"]
    return None, []


def notes_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    head = f"{row['date']}  {row['title']} — {row['company']}"
    if row.get("tag"):
        head += f"  [{row['tag']}]"
    if row["days_ago"] == 0:
        head += " — today"
    elif row["days_ago"] == 1:
        head += " — yesterday"
    elif row["days_ago"] > 1:
        head += f" — {row['days_ago']}d ago"
    else:
        head += f" — in {-row['days_ago']}d"
    return head + f"\n      {row['text']}"


def render_notes_markdown(data):
    """Human-readable note list: newest first, with a tag roll-up."""
    rows = data["notes"]
    summary = data["summary"]
    header = (f"Notes — {summary['total']} recorded")
    lines = [header]
    if summary["by_tag"]:
        lines.append("  " + ", ".join(
            f"{count} {tag}" for tag, count in summary["by_tag"].items()))
    if not rows:
        lines.append("No notes recorded yet — use "
                     "`notes --record --id APP-001 --date YYYY-MM-DD "
                     "--text 'What you learned'` to write one down.")
        return "\n".join(lines) + "\n"
    lines.append("")
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
