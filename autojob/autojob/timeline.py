"""Activity timeline: everything that happened, in one place, newest first.

``autojob timeline`` reads the history every other command already
records — applications, status moves, follow-ups, interviews, contacts,
offers — and lays them out as a single chronological feed across all
applications, instead of making you open each one to see what happened
when.

The honest rules, carried over from the modules that write the history:

- The timeline only ever shows what is already in the log. It invents
  nothing: every line comes from a history event, and the details on a
  line come from the record that event points at. An interview line
  names the round, the interviewer, the mode, and the duration exactly
  as recorded; an offer line states the base, bonus, and equity exactly
  as recorded. If a record is missing — a hand-edited log, an offer
  recorded before this version — the line falls back to the bare fact
  ("Offer recorded") rather than guessing the figures.
- Events are matched to their records, not assumed. An interview event
  carries its date and round label, so it is paired with the first
  still-unpaired round with that date and label; a contact event the
  same way by date and name. An offer event carries no figures of its
  own, so the n-th offer event on an application is the n-th offer in
  its offers list — they are appended in the same order by
  ``tracker.record_offer``, and the pairing is only ever used to
  display the figures, never to change them.
- A broken line never breaks the feed. An event with an unparseable
  date, or a kind the timeline does not know, is skipped, never fatal
  — a hand-edited log line must not hide everything else that
  happened. The same goes for a record that is not a dict, or a
  history that is not a list.
- Dates are the dates in the log, nothing else. ``--since`` and
  ``--until`` are inclusive bounds on the event date, so ``--until
  2026-11-09`` still shows the 9th. The default order is newest first
  — the thing you most often want is what happened lately — and
  ``--oldest`` flips it to chronological, log order within a day
  either way.
"""

from __future__ import annotations

import datetime
import json

KINDS = ("applied", "status", "followup", "interview", "contact", "offer")


class TimelineError(ValueError):
    """A timeline request that cannot be answered as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _fmt_amount(value):
    """A money figure as written: 130000 -> '130,000', 105000.5 stays."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:,}"


def _match_score(entry):
    score = entry.get("match_score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return score
    return None


def _summary_applied(entry):
    score = _match_score(entry)
    text = (f"Applied to {entry.get('title') or 'unknown role'} "
            f"at {entry.get('company') or 'unknown company'}")
    if score is not None:
        text += f" (match {score})"
    return text


def _summary_status(event):
    return f"Moved to {event.get('status')}"


def _summary_followup(event):
    text = "Follow-up sent"
    note = event.get("note")
    if isinstance(note, str) and note.strip():
        text += f" — {note.strip()}"
    return text


def _summary_interview(round_label, record):
    label = round_label or (record or {}).get("round")
    text = f"Interview: {label}" if label else "Interview"
    if not record:
        return text
    interviewer = record.get("interviewer")
    if isinstance(interviewer, str) and interviewer.strip():
        text += f" with {interviewer.strip()}"
    mode = record.get("mode")
    if isinstance(mode, str) and mode.strip():
        text += f" ({mode.strip()}"
        duration = record.get("duration_minutes")
        if (isinstance(duration, int) and not isinstance(duration, bool)):
            text += f", {duration} min"
        time = record.get("time")
        if isinstance(time, str) and time.strip():
            text += f", {time.strip()}"
        text += ")"
    else:
        duration = record.get("duration_minutes")
        if (isinstance(duration, int) and not isinstance(duration, bool)):
            text += f" ({duration} min)"
    return text


def _summary_contact(name, record):
    text = f"Spoke to {name}" if name else "Spoke to someone"
    if not record:
        return text
    role = record.get("role")
    if isinstance(role, str) and role.strip():
        text += f" ({role.strip()})"
    channel = record.get("channel")
    if isinstance(channel, str) and channel.strip():
        text += f" via {channel.strip()}"
    return text


def _summary_offer(record):
    if not record:
        return "Offer recorded"
    base = record.get("base")
    currency = record.get("currency")
    if not (isinstance(base, (int, float)) and not isinstance(base, bool)):
        return "Offer recorded"
    text = f"Offer recorded: {_fmt_amount(base)}"
    if isinstance(currency, str) and currency.strip():
        text += f" {currency.strip()}"
    period = record.get("period")
    if isinstance(period, str) and period.strip() and period != "annual":
        text += f" {period.strip()}"
    text += " base"
    bonus = record.get("bonus")
    if (isinstance(bonus, (int, float)) and not isinstance(bonus, bool)
            and bonus):
        text += f" + {_fmt_amount(bonus)} bonus"
    equity = record.get("equity")
    if (isinstance(equity, (int, float)) and not isinstance(equity, bool)
            and equity):
        text += f" + {_fmt_amount(equity)} equity"
        vesting = record.get("vesting_years")
        if (isinstance(vesting, int) and not isinstance(vesting, bool)):
            text += f" over {vesting}y"
    return text


def _note_of(kind, event, record):
    """A stored note worth showing under the line, or None.

    Follow-up notes live on the event itself; interview and contact
    notes live on the record. Offer notes are left out — an offer
    line already carries its figures.
    """
    if kind == "followup":
        note = event.get("note")
    elif kind in ("interview", "contact") and record:
        note = record.get("note") or record.get("notes")
    else:
        return None
    if isinstance(note, str) and note.strip():
        return note.strip()
    return None


def _resolve(entry, event, used_interviews, used_contacts, offer_index):
    """Turn one history event into a timeline row, or None to skip it.

    ``used_interviews`` / ``used_contacts`` are sets of record indexes
    already paired, so two rounds with the same date and label — or
    two touches with the same person on the same day — pair in log
    order instead of both pointing at the first record. ``offer_index``
    counts the offer events seen so far on this entry; the n-th offer
    event pairs with the n-th offer in the entry's offers list.
    """
    when = _parse_date(event.get("at"))
    if when is None:
        return None
    kind = event.get("kind")
    if kind is None and isinstance(event.get("status"), str):
        kind = "applied" if event["status"] == "applied" else "status"
    if kind not in KINDS:
        return None

    record = None
    if kind == "interview":
        rounds = entry.get("interviews") or []
        for i, candidate in enumerate(rounds):
            if i in used_interviews or not isinstance(candidate, dict):
                continue
            if (candidate.get("date") == event.get("at")
                    and candidate.get("round") == event.get("round")):
                record = candidate
                used_interviews.add(i)
                break
    elif kind == "contact":
        contacts = entry.get("contacts") or []
        for i, candidate in enumerate(contacts):
            if i in used_contacts or not isinstance(candidate, dict):
                continue
            if (candidate.get("date") == event.get("at")
                    and candidate.get("name") == event.get("name")):
                record = candidate
                used_contacts.add(i)
                break
    elif kind == "offer":
        offers = entry.get("offers") or []
        if 0 <= offer_index < len(offers) and isinstance(
                offers[offer_index], dict):
            record = offers[offer_index]

    if kind == "applied":
        summary = _summary_applied(entry)
    elif kind == "status":
        summary = _summary_status(event)
    elif kind == "followup":
        summary = _summary_followup(event)
    elif kind == "interview":
        summary = _summary_interview(event.get("round"), record)
    elif kind == "contact":
        summary = _summary_contact(event.get("name"), record)
    else:  # offer
        summary = _summary_offer(record)

    row = {
        "date": when.isoformat(),
        "kind": kind,
        "id": entry.get("id"),
        "company": entry.get("company"),
        "title": entry.get("title"),
        "summary": summary,
    }
    note = _note_of(kind, event, record)
    if note is not None:
        row["note"] = note
    return row, when


def collect_timeline(entries, today=None, since=None, until=None,
                     kinds=None, app_id=None, oldest=False):
    """Every recorded event, newest first (or oldest first with --oldest).

    Returns {"events": [...], "counts": {"total": n, "by_kind": {...}},
    "range": {"from": date, "to": date} or None, "order": ...}.
    ``since`` / ``until`` are ``datetime.date`` bounds, inclusive.
    ``kinds`` filters to those event kinds. ``app_id`` limits the feed
    to one application; an unknown id returns no events, and the CLI
    reports it. ``today`` is accepted for symmetry with the other
    list commands but the feed needs no reference date — it is
    ignored.
    """
    if kinds is not None:
        kinds = tuple(kinds)
    events = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if app_id is not None and entry.get("id") != app_id:
            continue
        history = entry.get("history")
        if not isinstance(history, list):
            continue
        used_interviews = set()
        used_contacts = set()
        offer_index = 0
        for event in history:
            if not isinstance(event, dict):
                continue
            resolved = _resolve(entry, event, used_interviews,
                                used_contacts, offer_index)
            if resolved is None:
                continue
            row, when = resolved
            if event.get("kind") == "offer":
                offer_index += 1
            if since is not None and when < since:
                continue
            if until is not None and when > until:
                continue
            if kinds is not None and row["kind"] not in kinds:
                continue
            events.append((when, row))

    # Newest first: date descending, and within a day the event
    # recorded later comes first — so the list reads like a diary
    # kept backwards. --oldest reverses both.
    if oldest:
        events.sort(key=lambda pair: pair[0])
        # Stable sort keeps log order within a day.
    else:
        events.sort(key=lambda pair: pair[0], reverse=True)
        # Reverse the within-day order too: stable sort on a reversed
        # list would keep it, so reverse the whole list after sorting
        # by date alone.
        events = _reverse_within_days(events)

    rows = [row for _when, row in events]
    by_kind = {}
    for row in rows:
        by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
    dates = [when for when, _row in events]
    return {
        "events": rows,
        "counts": {"total": len(rows), "by_kind": by_kind},
        "range": ({"from": min(dates).isoformat(),
                   "to": max(dates).isoformat()} if dates else None),
        "order": "oldest first" if oldest else "newest first",
    }


def _reverse_within_days(events):
    """Reverse each same-day run, so the latest-recorded event that day
    comes first. ``events`` arrives sorted by date descending."""
    out = []
    i = 0
    while i < len(events):
        j = i
        while j < len(events) and events[j][0] == events[i][0]:
            j += 1
        out.extend(reversed(events[i:j]))
        i = j
    return out


def timeline_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def render_timeline_markdown(data):
    """Human-readable feed, grouped by day."""
    events = data["events"]
    counts = data["counts"]
    lines = [f"Timeline — {counts['total']} event"
             + ("s" if counts["total"] != 1 else "")]
    if data["range"]:
        lines[0] += (f", {data['range']['from']} to {data['range']['to']}")
    lines[0] += f" ({data['order']})"
    if not events:
        lines.append("No events in this range — the log has nothing "
                     "recorded yet, or the filters left nothing out. "
                     "Try `timeline` on its own, or widen --since / --until.")
        return "\n".join(lines) + "\n"
    by_kind = counts["by_kind"]
    lines.append("  " + ", ".join(
        f"{count} {kind}" for kind, count in sorted(by_kind.items())))
    lines.append("")
    current_day = None
    for event in events:
        if event["date"] != current_day:
            current_day = event["date"]
            lines.append(current_day)
        lines.append(
            f"  - [{event['kind']}] {event['company'] or 'unknown company'}"
            f" — {event['title'] or 'unknown role'}"
            f" ({event['id'] or '?'}): {event['summary']}")
        if event.get("note"):
            lines.append(f"      note: {event['note']}")
    return "\n".join(lines) + "\n"
