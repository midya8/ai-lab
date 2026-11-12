"""Calendar export: turn the tracker into a .ics file you can import.

``autojob calendar`` builds one calendar over the next ``--days`` days
(default 14) from four things the tracker already knows:

- interview rounds in the window (from ``interview.py``);
- offer decision deadlines in the window (the latest offer per
  application, from ``offer.py``);
- posting closing dates in the window (the entry's ``deadline`` field,
  from ``deadlines.py``);
- follow-ups due as of today (from ``followup.py``), dated today —
  the action is due now, so the event is today, not the day the
  application went quiet.

The honest rules, carried over from the rest of autojob:

- Every event comes from a fact already in the log. Nothing is
  inferred: no interviewer is guessed, no time is invented, and a
  follow-up event is dated today because that is when the action is
  due, which the description says.
- Times are wall-clock times exactly as the user recorded them. The
  log stores no timezone, so the .ics uses floating local times (no
  ``TZID``, no ``Z``) — the event means "2:30pm wherever you are",
  which is what a user who typed ``--time 14:30`` meant. All-day
  events are ``VALUE=DATE`` with an exclusive end date, per RFC 5545.
- A round with a time but no recorded duration is shown as one hour,
  and the description says so — the hour is a display assumption, not
  a recorded fact. A round with no time at all is an all-day event.
- Deadlines and interviews on terminal applications (hired / rejected
  / withdrawn) are left out: there is nothing left to schedule or
  decide for them. Follow-ups already skip terminal applications.
- UIDs are deterministic (a hash of kind + application + date +
  summary), so exporting the same log twice produces the same file
  and importing it twice does not duplicate events.
"""

from __future__ import annotations

import datetime
import hashlib
import json

from .analytics import TERMINAL_STATUSES
from .followup import due_followups
from .interview import collect_interviews
from .offer import first_year_total, latest_offer

DEFAULT_DAYS = 14
DEFAULT_DURATION_MINUTES = 60  # shown when a time is recorded but no duration


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _parse_time(value):
    """A ``datetime.time`` for an HH:MM value, or None when unusable."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.time.fromisoformat(value)
    except ValueError:
        return None


def _uid(kind, app_id, date, summary):
    digest = hashlib.sha1(
        f"autojob|{kind}|{app_id}|{date}|{summary}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{digest}@autojob"


def _event(kind, app_id, company, title, date, summary, description,
           start=None, end=None, days_until=None):
    """One event dict; ``start``/``end`` are HH:MM strings or None.

    An event with no start is all-day on ``date``.
    """
    return {
        "kind": kind,
        "id": app_id,
        "company": company,
        "title": title,
        "date": date,
        "start": start,
        "end": end,
        "all_day": start is None,
        "summary": summary,
        "description": description,
        "days_until": days_until,
        "uid": _uid(kind, app_id, date, summary),
    }


def _interview_events(entries, today, end):
    """Upcoming rounds in the window, soonest first.

    Rounds on terminal applications are skipped — there is nothing
    left to schedule for them.
    """
    terminal = {e.get("id") for e in entries
                if e.get("status") in TERMINAL_STATUSES}
    rows = collect_interviews(entries, today=today)["upcoming"]
    events = []
    for row in rows:
        if row["id"] in terminal:
            continue
        when = _parse_date(row["date"])
        if when is None or when > end:
            continue
        start = _parse_time(row.get("time"))
        parts = [f"{row['round']} — {row['title']} at {row['company']}"]
        description = [f"Interview: {row['round']} for the {row['title']} "
                       f"role at {row['company']}."]
        if row.get("interviewer"):
            description.append(f"Interviewer: {row['interviewer']}.")
        if row.get("mode"):
            description.append(f"Mode: {row['mode']}.")
        if row.get("notes"):
            description.append(f"Notes: {row['notes']}")
        if start is None:
            summary = f"Interview: {parts[0]}"
            events.append(_event("interview", row["id"], row["company"],
                                 row["title"], row["date"], summary,
                                 " ".join(description),
                                 days_until=row["days_until"]))
            continue
        minutes = row.get("duration_minutes")
        if minutes is None:
            minutes = DEFAULT_DURATION_MINUTES
            description.append("Duration not recorded — shown as 1 hour.")
        else:
            description.append(f"Duration: {minutes} minutes.")
        start_dt = datetime.datetime.combine(when, start)
        end_dt = start_dt + datetime.timedelta(minutes=minutes)
        summary = (f"Interview {start.strftime('%H:%M')}: {parts[0]}")
        events.append(_event("interview", row["id"], row["company"],
                             row["title"], row["date"], summary,
                             " ".join(description),
                             start=start.strftime("%H:%M"),
                             end=end_dt.strftime("%H:%M"),
                             days_until=row["days_until"]))
    return events


def _offer_deadline_events(entries, today, end):
    """Decision deadlines in the window, one per application.

    Uses the latest recorded offer; the description states the base
    as recorded and the first-year total the offer comparison uses.
    Deadlines already passed are outside the window by construction.
    """
    events = []
    for entry in entries:
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        offer = latest_offer(entry)
        if offer is None:
            continue
        deadline = _parse_date(offer.get("deadline"))
        if deadline is None or deadline < today or deadline > end:
            continue
        total, equity_counted = first_year_total(offer)
        summary = (f"Offer deadline: {entry.get('title')} at "
                   f"{entry.get('company')}")
        description = [
            f"Decision deadline for the {entry.get('title')} offer at "
            f"{entry.get('company')}.",
            f"Base {offer.get('currency')} {offer.get('base_annual'):,.0f} "
            f"annual ({offer.get('currency')} {offer.get('base'):,.0f} "
            f"{offer.get('period') or 'annual'} as stated).",
            f"First-year total {offer.get('currency')} {total:,.0f}"
            + (" (equity counted)" if equity_counted
               else " (equity not counted — no vesting schedule)")
            + ".",
        ]
        if offer.get("note"):
            description.append(f"Note: {offer['note']}")
        events.append(_event("offer_deadline", entry.get("id"),
                             entry.get("company"), entry.get("title"),
                             offer["deadline"], summary,
                             " ".join(description),
                             days_until=(deadline - today).days))
    return events


def _posting_deadline_events(entries, today, end):
    """Posting closing dates in the window, one per application.

    Uses the entry's ``deadline`` field (see deadlines.py). A closing
    date already passed is outside the window by construction, so an
    overdue posting never exports — the list command is where overdue
    dates are watched. Deadlines on terminal applications are left
    out, the same rule as interviews and offer deadlines.
    """
    events = []
    for entry in entries:
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        deadline = _parse_date(entry.get("deadline"))
        if deadline is None or deadline < today or deadline > end:
            continue
        summary = (f"Applications close: {entry.get('title')} at "
                   f"{entry.get('company')}")
        description = [
            f"Applications for the {entry.get('title')} role at "
            f"{entry.get('company')} close on {entry.get('deadline')} "
            f"(status {entry.get('status')}, applied "
            f"{entry.get('applied_at')}).",
            "Recorded with `deadlines --record`; see `deadlines` for "
            "the full list, including overdue ones.",
        ]
        events.append(_event("posting_deadline", entry.get("id"),
                             entry.get("company"), entry.get("title"),
                             entry["deadline"], summary,
                             " ".join(description),
                             days_until=(deadline - today).days))
    return events


def _followup_events(entries, today):
    """Applications due a follow-up as of today, each dated today."""
    events = []
    for item in due_followups(entries, today=today):
        summary = f"Follow up: {item['title']} at {item['company']}"
        description = [
            f"Follow-up due for the {item['title']} application at "
            f"{item['company']} (status {item['status']}, applied "
            f"{item['applied_at']}).",
            f"Waiting {item['days_waiting']} days (wait "
            f"{item['wait_days']}); "
            f"{item['followups_sent']} follow-up(s) already sent.",
            "Dated today because the follow-up is due now, not because "
            "the application went quiet today.",
        ]
        events.append(_event("followup", item["id"], item["company"],
                             item["title"], today.isoformat(), summary,
                             " ".join(description), days_until=0))
    return events


def build_calendar(entries, today=None, days=DEFAULT_DAYS, wait_days=None):
    """All events in [today, today + days], plus due follow-ups.

    Returns {"events": [...], "window": {...}, "counts": {...}}.
    ``today`` is a ``datetime.date`` (defaults to the real current
    date; pass an explicit date in tests). ``days`` must be 0 or more;
    0 means today only. Events sort by date, then start time, then
    kind (interviews first), then company and title.
    """
    if today is None:
        today = datetime.date.today()
    if days is None or days < 0:
        raise ValueError(f"days must be 0 or more, got {days!r}")
    end = today + datetime.timedelta(days=days)
    events = []
    events.extend(_interview_events(entries, today, end))
    events.extend(_offer_deadline_events(entries, today, end))
    events.extend(_posting_deadline_events(entries, today, end))
    events.extend(_followup_events(entries, today))
    kind_order = {"interview": 0, "offer_deadline": 1,
                  "posting_deadline": 2, "followup": 3}
    events.sort(key=lambda e: (e["date"], e["start"] or "",
                               kind_order[e["kind"]],
                               (e["company"] or "").lower(),
                               (e["title"] or "").lower()))
    return {
        "events": events,
        "window": {"from": today.isoformat(), "to": end.isoformat(),
                   "days": days},
        "counts": {
            "events": len(events),
            "interviews": sum(1 for e in events if e["kind"] == "interview"),
            "offer_deadlines": sum(1 for e in events
                                   if e["kind"] == "offer_deadline"),
            "posting_deadlines": sum(1 for e in events
                                     if e["kind"] == "posting_deadline"),
            "followups": sum(1 for e in events if e["kind"] == "followup"),
        },
    }


def _escape_text(value):
    """RFC 5545 TEXT escaping: backslash, semicolon, comma, newline."""
    return (value.replace("\\", "\\\\")
                 .replace(";", "\\;")
                 .replace(",", "\\,")
                 .replace("\n", "\\n"))


def _fold_line(line):
    """Fold one logical line to 75-octet content lines (RFC 5545).

    Folding never splits a multi-byte UTF-8 character; continuation
    lines start with a single space, so unfolding (remove CRLF + one
    space) restores the original line exactly.
    """
    encoded = line.encode("utf-8")
    if len(encoded) <= 75:
        return [line]
    lines = []
    start = 0
    first = True
    while start < len(encoded):
        # A continuation line spends one octet on its leading space.
        width = 75 if first else 74
        limit = min(start + width, len(encoded))
        # Back off to a UTF-8 character boundary.
        while (limit > start and limit < len(encoded)
               and (encoded[limit] & 0xC0) == 0x80):
            limit -= 1
        if limit == start:  # a single character wider than the width
            limit = start + width
        chunk = encoded[start:limit].decode("utf-8")
        lines.append(chunk if first else " " + chunk)
        start = limit
        first = False
    return lines


def render_ics(data):
    """The calendar as an .ics file (CRLF line endings, folded lines).

    DTSTAMP is the window's start date at midnight UTC — a file
    generation stamp, not a claim about when anything happens.
    Timed events use floating local times: the log stores no timezone,
    so none is invented.
    """
    window_from = data["window"]["from"].replace("-", "")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0",
             "PRODID:-//autojob//EN", "CALSCALE:GREGORIAN",
             "METHOD:PUBLISH"]
    for event in data["events"]:
        date = event["date"].replace("-", "")
        lines.append("BEGIN:VEVENT")
        lines.append(f"UID:{event['uid']}")
        lines.append(f"DTSTAMP:{window_from}T000000Z")
        if event["all_day"]:
            end = (datetime.date.fromisoformat(event["date"])
                   + datetime.timedelta(days=1))
            lines.append(f"DTSTART;VALUE=DATE:{date}")
            lines.append(f"DTEND;VALUE=DATE:{end.strftime('%Y%m%d')}")
        else:
            start = event["start"].replace(":", "")
            end = event["end"].replace(":", "")
            lines.append(f"DTSTART:{date}T{start}00")
            lines.append(f"DTEND:{date}T{end}00")
        lines.append(f"SUMMARY:{_escape_text(event['summary'])}")
        lines.append(f"DESCRIPTION:{_escape_text(event['description'])}")
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    folded = []
    for line in lines:
        folded.extend(_fold_line(line))
    return "\r\n".join(folded) + "\r\n"


def calendar_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"
