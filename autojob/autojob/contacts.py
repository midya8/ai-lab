"""Contacts: the people behind each application.

``contacts --record`` stores a person on a tracker entry — name, role,
channel, the date you first reached them (or last spoke to them), and a
note. ``contacts`` on its own lists everyone, quietest first, so the
people you have not spoken to in a while are the first thing you see.

The honest rules, carried over from interview.py and offer.py:

- Everything recorded is a fact the user supplies. The name, the role
  ("Recruiter", "Hiring manager", "Referrer"), the channel, the date,
  and the note are all the user's own words — this module never
  generates text about a person, so it cannot invent a contact, a
  conversation, or a relationship. There is deliberately no email or
  phone field: a name plus a channel ("LinkedIn", "email") is enough
  to know who someone is and how you reached them, and the log never
  holds contact details it does not need.
- A contact is a person, not a status: recording one does not move the
  application's status — that stays explicit through ``status``, the
  same rule as offers and interviews. Recording is refused on a
  terminal application (hired / rejected / withdrawn): there is nobody
  left to talk to once an application is closed.
- Recording the same person again is a new touch, not a duplicate:
  you spoke to them again, so the list shows one row per person with
  the latest touch, how many touches there have been, and when you
  first reached them. Two records with the same name on the same
  application are the same person — the match is case-insensitive,
  and the row keeps the name as most recently recorded.
- "Quiet" is about the person, not the application. A contact is
  quiet when you have not spoken to *them* for longer than
  ``--quiet-days`` (default 14) — an interview with someone else, or
  an offer arriving, does not count as speaking to this person.
  Exactly N days since the last touch is not quiet yet, the same
  boundary the stale check and the follow-up wait use. Contacts on
  terminal applications are left out of the list: they need no nudge.
- The date is required and is never defaulted. Defaulting it to today
  would silently claim you spoke to someone today when you may only
  be writing down a name you met last week.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES

CHANNELS = ("email", "linkedin", "phone", "in-person", "other")

DEFAULT_QUIET_DAYS = 14


class ContactError(ValueError):
    """A contact that cannot be recorded as stated."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_contact(data, today=None):
    """Validate a contact dict and return the cleaned form.

    Every problem is reported at once, like profile, offer, and
    interview validation. ``today`` defaults to the real current date;
    pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    errors = []
    data = data or {}

    name = data.get("name")
    if name is None or (isinstance(name, str) and not name.strip()):
        errors.append("name is required")
    elif not isinstance(name, str):
        errors.append(f"name must be text, got {name!r}")

    role = data.get("role")
    if role is not None and not isinstance(role, str):
        errors.append(f"role must be text, got {role!r}")

    channel = data.get("channel")
    if channel is not None and channel not in CHANNELS:
        errors.append(f"channel must be one of {', '.join(CHANNELS)}, "
                      f"got {channel!r}")

    date = data.get("date")
    if date is None or date == "":
        errors.append("date is required (when you first reached them, "
                      "or last spoke to them)")
    else:
        try:
            datetime.date.fromisoformat(date)
        except (ValueError, TypeError):
            errors.append(f"date must be YYYY-MM-DD, got {date!r}")

    note = data.get("note")
    if note is not None and not isinstance(note, str):
        errors.append(f"note must be text, got {note!r}")

    if errors:
        raise ContactError("; ".join(errors))

    return {
        "name": name.strip(),
        "role": role.strip() if isinstance(role, str) and role.strip()
                else None,
        "channel": channel,
        "date": date,
        "note": note,
        "recorded_at": today,
    }


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _row(entry, records, today, quiet_days):
    """One row per person: latest touch, first touch, touch count.

    ``records`` are that person's contact dicts in log order. A record
    with an unparseable date contributes nothing — a hand-edited log
    line must not break the whole list. The row keeps the role and
    channel from the latest dated record, falling back to the latest
    record of any kind.
    """
    dated = [(rec, _parse_date(rec.get("date"))) for rec in records]
    dated = [(rec, when) for rec, when in dated if when is not None]
    if not dated:
        return None
    dated.sort(key=lambda pair: pair[1])
    first = dated[0][0]
    latest, last_touch = dated[-1]
    role = latest.get("role")
    channel = latest.get("channel")
    note = latest.get("note")
    if role is None or channel is None or note is None:
        for rec, _when in reversed(dated):
            if role is None:
                role = rec.get("role")
            if channel is None:
                channel = rec.get("channel")
            if note is None:
                note = rec.get("note")
            if (role is not None and channel is not None
                    and note is not None):
                break
    days_since = (today - last_touch).days
    return {
        "id": entry.get("id"),
        "company": entry.get("company"),
        "title": entry.get("title"),
        "status": entry.get("status"),
        "name": latest.get("name") or first.get("name"),
        "role": role,
        "channel": channel,
        "first_contact": first.get("date"),
        "last_contact": latest.get("date"),
        "touches": len(records),
        "note": note,
        "days_since_touch": days_since,
        "quiet": days_since > quiet_days,
    }


def collect_contacts(entries, today=None, quiet_days=DEFAULT_QUIET_DAYS):
    """Everyone on a live application, quietest first.

    Returns {"contacts": [...], "counts": {"total": n, "quiet": n}}.
    Entries without a ``contacts`` list — everything logged before
    v0.13 — simply contribute nobody. Contacts on terminal
    applications are left out: they need no nudge. ``today`` is a
    ``datetime.date`` (defaults to the real current date; pass an
    explicit date in tests).
    """
    if today is None:
        today = datetime.date.today()
    if quiet_days is None:
        quiet_days = DEFAULT_QUIET_DAYS
    rows = []
    for entry in entries:
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        people = {}
        for contact in entry.get("contacts") or []:
            if not isinstance(contact, dict):
                continue
            name = contact.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            people.setdefault(name.strip().lower(), []).append(contact)
        for records in people.values():
            row = _row(entry, records, today, quiet_days)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: (-row["days_since_touch"],
                               (row["company"] or "").lower(),
                               (row["title"] or "").lower(),
                               (row["name"] or "").lower()))
    return {
        "contacts": rows,
        "counts": {
            "total": len(rows),
            "quiet": sum(1 for row in rows if row["quiet"]),
        },
    }


def contacts_for(entries, app_id, today=None,
                 quiet_days=DEFAULT_QUIET_DAYS):
    """All contacts for one application, quietest first.

    Returns the entry plus its rows, or (None, []) when the id is
    unknown. The entry is returned even when it holds no contacts,
    so the caller can tell "known application, nobody recorded" apart
    from "no such application".
    """
    if today is None:
        today = datetime.date.today()
    if quiet_days is None:
        quiet_days = DEFAULT_QUIET_DAYS
    for entry in entries:
        if entry.get("id") == app_id:
            people = {}
            for contact in entry.get("contacts") or []:
                if not isinstance(contact, dict):
                    continue
                name = contact.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                people.setdefault(name.strip().lower(), []).append(contact)
            rows = []
            for records in people.values():
                row = _row(entry, records, today, quiet_days)
                if row is not None:
                    rows.append(row)
            rows.sort(key=lambda row: (-row["days_since_touch"],
                                       (row["name"] or "").lower()))
            return entry, rows
    return None, []


def contacts_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    parts = [f"{row['name']}  {row['title']} — {row['company']}"]
    if row.get("role"):
        parts.append(f"({row['role']})")
    if row.get("channel"):
        parts.append(f"via {row['channel']}")
    if row["days_since_touch"] < 0:
        parts.append(f"— on {row['last_contact']} "
                     f"(in {-row['days_since_touch']}d)")
    elif row["days_since_touch"] == 0:
        parts.append("— spoke today")
    elif row["days_since_touch"] == 1:
        parts.append("— spoke yesterday")
    else:
        parts.append(f"— {row['days_since_touch']}d since last spoke")
    if row.get("touches", 1) > 1:
        parts.append(f"({row['touches']} touches, "
                     f"first {row['first_contact']})")
    line = " ".join(parts)
    if row.get("quiet"):
        line += "  [quiet]"
    if row.get("note"):
        line += f"\n      note: {row['note']}"
    return line


def render_contacts_markdown(data):
    """Human-readable contacts list: quietest first."""
    contacts = data["contacts"]
    counts = data["counts"]
    lines = [f"Contacts — {counts['total']} people"
             + (f", {counts['quiet']} quiet" if counts["quiet"] else "")]
    if not contacts:
        lines.append("No contacts recorded yet — use "
                     "`contacts --record --id APP-001 --name 'Sam Chen' "
                     "--role Recruiter --channel linkedin "
                     "--date YYYY-MM-DD` to record one.")
        return "\n".join(lines) + "\n"
    lines.append("")
    for row in contacts:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
