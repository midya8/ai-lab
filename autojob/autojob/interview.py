"""Interview rounds log: keep every scheduled interview in one place.

``interview --record`` stores a round on a tracker entry (date, round
label, interviewer, mode, duration, notes). ``interview`` on its own
lists what is coming up and what already happened, so the week ahead is
one command instead of a hunt through inboxes.

The honest rules, carried over from offer.py and followup.py:

- Everything recorded is a fact the user supplies. The round label
  ("Recruiter screen", "Hiring manager", "Panel"), the interviewer
  name, the mode, and the notes are all the user's own words — this
  module never generates text about an interview, so it cannot invent
  an interviewer, a question, or an outcome.
- A round is a date, not a status: recording one does not move the
  application's status — that stays explicit through ``status``, the
  same rule as offers. An application can hold several rounds on the
  same day (two interviews in one day is a real week), so rounds are
  never deduplicated.
- "Today" counts as upcoming, with 0 days to go. A round dated before
  today is past; the past list is most recent first, the upcoming list
  soonest first. Dates are compared as dates, never as datetimes, so
  no timezone can move a round to the wrong list.
- Recording is refused on a terminal application (hired / rejected /
  withdrawn): there is no interview to schedule once an application
  is closed.
"""

from __future__ import annotations

import datetime
import json

MODES = ("phone", "video", "onsite")


class InterviewError(ValueError):
    """An interview round that cannot be recorded as stated."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_interview(data, today=None):
    """Validate an interview round dict and return the cleaned form.

    Every problem is reported at once, like profile and offer
    validation. ``today`` defaults to the real current date; pass an
    explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    errors = []
    data = data or {}

    date = data.get("date")
    if date is None or date == "":
        errors.append("date is required")
    else:
        try:
            datetime.date.fromisoformat(date)
        except (ValueError, TypeError):
            errors.append(f"date must be YYYY-MM-DD, got {date!r}")

    round_label = data.get("round")
    if round_label is None or (isinstance(round_label, str)
                               and not round_label.strip()):
        errors.append("round is required (e.g. 'Recruiter screen')")
    elif not isinstance(round_label, str):
        errors.append(f"round must be text, got {round_label!r}")

    interviewer = data.get("interviewer")
    if interviewer is not None and not isinstance(interviewer, str):
        errors.append(f"interviewer must be text, got {interviewer!r}")

    mode = data.get("mode")
    if mode is not None and mode not in MODES:
        errors.append(f"mode must be one of {', '.join(MODES)}, "
                      f"got {mode!r}")

    duration = data.get("duration_minutes")
    if duration is not None:
        if not _is_int(duration) or duration <= 0:
            errors.append(f"duration_minutes must be a whole number above 0, "
                          f"got {duration!r}")

    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append(f"notes must be text, got {notes!r}")

    if errors:
        raise InterviewError("; ".join(errors))

    return {
        "date": date,
        "round": round_label.strip(),
        "interviewer": interviewer.strip() if isinstance(interviewer, str)
                       and interviewer.strip() else None,
        "mode": mode,
        "duration_minutes": duration,
        "notes": notes,
        "recorded_at": today,
    }


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _rows(entries, today, upcoming):
    """(rows, n_rounds) for one side of the today boundary.

    A row carries the facts the CLI and the report need: the entry's
    identity, the round as recorded, and how far away it is. Rows with
    an unparseable date are skipped, never fatal — a hand-edited log
    line must not break the whole list.
    """
    rows = []
    for entry in entries:
        for interview in entry.get("interviews") or []:
            when = _parse_date(interview.get("date"))
            if when is None:
                continue
            delta = (when - today).days
            if upcoming and delta < 0:
                continue
            if not upcoming and delta >= 0:
                continue
            rows.append({
                "id": entry.get("id"),
                "company": entry.get("company"),
                "title": entry.get("title"),
                "status": entry.get("status"),
                "date": interview["date"],
                "round": interview.get("round"),
                "interviewer": interview.get("interviewer"),
                "mode": interview.get("mode"),
                "duration_minutes": interview.get("duration_minutes"),
                "notes": interview.get("notes"),
                "days_until": delta if upcoming else None,
                "days_ago": -delta if not upcoming else None,
            })
    if upcoming:
        rows.sort(key=lambda row: (row["date"],
                                   (row["company"] or "").lower(),
                                   (row["title"] or "").lower(),
                                   (row["round"] or "").lower()))
    else:
        rows.sort(key=lambda row: (row["date"],
                                   (row["company"] or "").lower(),
                                   (row["title"] or "").lower(),
                                   (row["round"] or "").lower()),
                  reverse=True)
    return rows


def collect_interviews(entries, today=None):
    """Split every recorded round into upcoming and past.

    Returns {"upcoming": [...], "past": [...], "counts": {...}}.
    ``today`` is a ``datetime.date`` (defaults to the real current
    date; pass an explicit date in tests). Entries without an
    ``interviews`` list — everything logged before v0.11 — simply
    contribute no rounds.
    """
    if today is None:
        today = datetime.date.today()
    upcoming = _rows(entries, today, upcoming=True)
    past = _rows(entries, today, upcoming=False)
    return {
        "upcoming": upcoming,
        "past": past,
        "counts": {
            "upcoming": len(upcoming),
            "past": len(past),
            "today": sum(1 for row in upcoming if row["days_until"] == 0),
        },
    }


def interviews_for(entries, app_id, today=None):
    """All rounds for one application, soonest first (past last).

    Returns the entry plus its rounds, or (None, []) when the id is
    unknown. The split into upcoming and past reuses the same
    boundary as collect_interviews: today is upcoming.
    """
    if today is None:
        today = datetime.date.today()
    for entry in entries:
        if entry.get("id") == app_id:
            rounds = []
            for interview in entry.get("interviews") or []:
                when = _parse_date(interview.get("date"))
                if when is None:
                    continue
                delta = (when - today).days
                rounds.append({
                    "id": entry.get("id"),
                    "company": entry.get("company"),
                    "title": entry.get("title"),
                    "status": entry.get("status"),
                    "date": interview["date"],
                    "round": interview.get("round"),
                    "interviewer": interview.get("interviewer"),
                    "mode": interview.get("mode"),
                    "duration_minutes": interview.get("duration_minutes"),
                    "notes": interview.get("notes"),
                    "days_until": delta if delta >= 0 else None,
                    "days_ago": -delta if delta < 0 else None,
                })
            rounds.sort(key=lambda row: (row["date"],
                                         (row["round"] or "").lower()))
            return entry, rounds
    return None, []


def interview_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    parts = [f"{row['date']}  {row['title']} — {row['company']}  "
             f"{row['round']}"]
    if row.get("interviewer"):
        parts.append(f"with {row['interviewer']}")
    if row.get("mode"):
        parts.append(f"({row['mode']})")
    if row.get("duration_minutes"):
        parts.append(f"{row['duration_minutes']} min")
    if row.get("days_until") == 0:
        parts.append("— today")
    elif row.get("days_until") is not None:
        parts.append(f"— in {row['days_until']}d")
    elif row.get("days_ago") is not None:
        if row["days_ago"] == 1:
            parts.append("— yesterday")
        else:
            parts.append(f"— {row['days_ago']}d ago")
    line = " ".join(parts)
    if row.get("notes"):
        line += f"\n      notes: {row['notes']}"
    return line


def render_interview_markdown(data):
    """Human-readable interview schedule: upcoming first, then past."""
    counts = data["counts"]
    lines = [f"Interviews — {counts['upcoming']} upcoming"
             + (f" ({counts['today']} today)" if counts["today"] else "")
             + f", {counts['past']} past"]
    upcoming = data["upcoming"]
    past = data["past"]
    if not upcoming and not past:
        lines.append("No interviews recorded yet — use "
                     "`interview --record --id APP-001 --date YYYY-MM-DD "
                     "--round 'Hiring manager'` to record one.")
        return "\n".join(lines) + "\n"
    if upcoming:
        lines.append("")
        lines.append("Upcoming:")
        for row in upcoming:
            lines.append(f"  {_fmt_row(row)}")
    if past:
        lines.append("")
        lines.append("Past:")
        for row in past:
            lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
