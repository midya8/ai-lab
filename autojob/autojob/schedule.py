"""Interview schedule: the week ahead as a calendar, with clashes flagged.

``autojob schedule`` lays out the dated commitments the log already
holds — interview rounds, offer decision deadlines, posting closing
dates — over the next ``--days`` days (default 14), one day at a
time, and flags the days that do not work: two interviews whose
times overlap, two interviews with less than a 30-minute gap between
them, a day carrying three things at once, and a round with no start
time recorded, which cannot be checked for overlaps at all.

The honest rules, carried over from ``calendar`` and ``audit``:

- Every item comes from a fact already in the log. Nothing is
  inferred: no interviewer is guessed, no time is invented, and a
  round with a time but no recorded duration is shown as one hour —
  the same display assumption ``calendar`` uses — with the item
  marked ``duration_assumed`` so the assumption is visible.
- Times are wall-clock times exactly as recorded, compared as times
  on the same date; no timezone is stored and none is invented.
- A round that ends exactly when the next one starts is not an
  overlap — the two do not collide — but a gap of zero minutes is
  still tighter than the 30-minute buffer, so it is flagged tight,
  not clean. Exactly 30 minutes between two rounds is not tight:
  the buffer is 30 minutes, and a full buffer is enough.
- A busy day is a day with three or more items, or with two or more
  all-day items. Two timed interviews on one day with a decent gap
  between them is a full day, not a busy one — the flag is about
  overload, not about having two things to do.
- Terminal applications (hired / rejected / withdrawn) are left
  out, the same rule as ``calendar``: there is nothing left to
  schedule or decide for them. A rejected application's interview
  cannot clash with anything.
- Dates and times that cannot be parsed are skipped, never fatal —
  the same rule as every other command. A round whose recorded time
  is not a HH:MM time is treated as untimed and flagged, not
  dropped silently.
- The command is read-only: it records nothing and never touches
  the log, and it exits 0 even when it finds a clash — it is a
  report, the same as ``audit``.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES
from .interview import collect_interviews
from .offer import first_year_total, latest_offer

SCHEDULE_KINDS = ("interview", "offer_deadline", "posting_deadline")

DEFAULT_DAYS = 14
DEFAULT_DURATION_MINUTES = 60  # shown when a time is recorded but no duration
TIGHT_GAP_MINUTES = 30  # less than this between two rounds is tight
_KIND_ORDER = {"interview": 0, "offer_deadline": 1, "posting_deadline": 2}
_SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}


class ScheduleError(ValueError):
    """A bad --days, --kind, or window for the schedule."""


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


def _minutes(time_value):
    return time_value.hour * 60 + time_value.minute


def _hhmm(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _item(kind, entry, date, start, end, detail, days_until,
          duration_assumed=False):
    return {
        "kind": kind,
        "id": entry.get("id"),
        "company": entry.get("company"),
        "title": entry.get("title"),
        "date": date,
        "start": start,
        "end": end,
        "all_day": start is None,
        "duration_assumed": duration_assumed,
        "detail": detail,
        "days_until": days_until,
    }


def _interview_items(entries, today, end):
    """Upcoming rounds in the window, live applications only."""
    terminal = {e.get("id") for e in entries
                if isinstance(e, dict)
                and e.get("status") in TERMINAL_STATUSES}
    rows = collect_interviews(
        [e for e in entries if isinstance(e, dict)],
        today=today)["upcoming"]
    items = []
    for row in rows:
        if row["id"] in terminal:
            continue
        when = _parse_date(row["date"])
        if when is None or when > end:
            continue
        start = _parse_time(row.get("time"))
        detail = [f"{row['round']} — {row['title']} at {row['company']}"]
        if row.get("interviewer"):
            detail.append(f"with {row['interviewer']}")
        if row.get("mode"):
            detail.append(f"({row['mode']})")
        if start is None:
            items.append(_item("interview", row, row["date"], None, None,
                               " ".join(detail), row["days_until"]))
            continue
        minutes = row.get("duration_minutes")
        assumed = False
        if not isinstance(minutes, int) or isinstance(minutes, bool) \
                or minutes <= 0:
            minutes = DEFAULT_DURATION_MINUTES
            assumed = True
        start_minutes = _minutes(start)
        end_minutes = start_minutes + minutes
        if end_minutes > 24 * 60:
            end_minutes = 24 * 60
        detail.append(f"{_hhmm(start_minutes)}–{_hhmm(end_minutes)}")
        if assumed:
            detail.append("(duration not recorded — shown as 1 hour)")
        items.append(_item("interview", row, row["date"],
                           _hhmm(start_minutes), _hhmm(end_minutes),
                           " ".join(detail), row["days_until"],
                           duration_assumed=assumed))
    return items


def _offer_deadline_items(entries, today, end):
    """Decision deadlines in the window, one per live application."""
    items = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        offer = latest_offer(entry)
        if offer is None:
            continue
        deadline = _parse_date(offer.get("deadline"))
        if deadline is None or deadline < today or deadline > end:
            continue
        total, equity_counted = first_year_total(offer)
        detail = (f"Decision due — {entry.get('title')} at "
                  f"{entry.get('company')} "
                  f"({offer.get('currency')} "
                  f"{offer.get('base_annual'):,.0f} base annual, "
                  f"first year {offer.get('currency')} {total:,.0f}"
                  + (" — equity counted" if equity_counted
                     else " — equity not counted")
                  + ")")
        items.append(_item("offer_deadline", entry, offer["deadline"],
                           None, None, detail, (deadline - today).days))
    return items


def _posting_deadline_items(entries, today, end):
    """Posting closing dates in the window, live applications only."""
    items = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        deadline = _parse_date(entry.get("deadline"))
        if deadline is None or deadline < today or deadline > end:
            continue
        detail = (f"Applications close — {entry.get('title')} at "
                  f"{entry.get('company')}")
        items.append(_item("posting_deadline", entry, entry["deadline"],
                           None, None, detail, (deadline - today).days))
    return items


def _day_flags(items):
    """Flags for one day's items: conflict, tight, busy, time-not-set.

    Conflicts are every overlapping pair of timed interviews; tight
    gaps are checked between adjacent timed interviews only, so a
    packed day reports each gap once instead of once per pair.
    """
    flags = []
    timed = [item for item in items if not item["all_day"]]
    untimed = [item for item in items
               if item["all_day"] and item["kind"] == "interview"]
    for item in untimed:
        flags.append({
            "severity": "info",
            "kind": "time-not-set",
            "date": item["date"],
            "detail": (f"{item['detail'].split(' — ')[0]} at "
                       f"{item['company']} has no usable start time "
                       "recorded — overlaps cannot be checked"),
            "minutes": None,
            "items": [item],
        })
    timed.sort(key=lambda i: (i["start"], i["end"],
                              (i["company"] or "").lower(),
                              (i["title"] or "").lower()))
    for i, first in enumerate(timed):
        for second in timed[i + 1:]:
            if second["start"] >= first["end"]:
                break
            overlap_start = second["start"]
            overlap_end = min(first["end"], second["end"])
            overlap = (_minutes(_parse_time(overlap_end))
                       - _minutes(_parse_time(overlap_start)))
            flags.append({
                "severity": "error",
                "kind": "conflict",
                "date": first["date"],
                "detail": (f"Interviews overlap "
                           f"{overlap_start}–{overlap_end} "
                           f"({overlap} min): {first['detail']} and "
                           f"{second['detail']}"),
                "minutes": overlap,
                "items": [first, second],
            })
    for first, second in zip(timed, timed[1:]):
        if second["start"] < first["end"]:
            continue  # an overlap is a conflict, not also tight
        gap = (_minutes(_parse_time(second["start"]))
               - _minutes(_parse_time(first["end"])))
        if gap < TIGHT_GAP_MINUTES:
            flags.append({
                "severity": "warn",
                "kind": "tight",
                "date": first["date"],
                "detail": (f"Only {gap} min between {first['detail']} "
                           f"and {second['detail']} — less than the "
                           f"{TIGHT_GAP_MINUTES}-minute buffer"),
                "minutes": gap,
                "items": [first, second],
            })
    all_day = [item for item in items if item["all_day"]]
    if len(items) >= 3:
        flags.append({
            "severity": "info",
            "kind": "busy",
            "date": items[0]["date"],
            "detail": f"{len(items)} things on this day",
            "minutes": None,
            "items": list(items),
        })
    elif len(all_day) >= 2:
        flags.append({
            "severity": "info",
            "kind": "busy",
            "date": items[0]["date"],
            "detail": f"{len(all_day)} all-day deadlines on this day",
            "minutes": None,
            "items": list(all_day),
        })
    return flags


def collect_schedule(entries, today=None, days=DEFAULT_DAYS, kinds=None):
    """Dated commitments over [today, today + days], grouped by day.

    Returns {"days": [...], "flags": [...], "counts": {...},
    "window": {...}}. ``today`` is a ``datetime.date`` (defaults to
    the real current date; pass an explicit date in tests). ``days``
    must be 0 or more; 0 means today only. ``kinds`` narrows the
    items to the named kinds (repeatable on the CLI). Unknown kinds
    or a negative window raise ScheduleError.
    """
    if today is None:
        today = datetime.date.today()
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        raise ScheduleError(f"--days must be 0 or more, got {days!r}")
    if kinds is not None:
        bad = [kind for kind in kinds if kind not in SCHEDULE_KINDS]
        if bad:
            raise ScheduleError(f"unknown --kind {bad[0]!r}; one of: "
                                f"{', '.join(SCHEDULE_KINDS)}")
        wanted = set(kinds)
    else:
        wanted = set(SCHEDULE_KINDS)
    end = today + datetime.timedelta(days=days)

    items = []
    if "interview" in wanted:
        items.extend(_interview_items(entries, today, end))
    if "offer_deadline" in wanted:
        items.extend(_offer_deadline_items(entries, today, end))
    if "posting_deadline" in wanted:
        items.extend(_posting_deadline_items(entries, today, end))
    items.sort(key=lambda i: (i["date"], i["start"] or "",
                              _KIND_ORDER[i["kind"]],
                              (i["company"] or "").lower(),
                              (i["title"] or "").lower()))

    by_day = {}
    for item in items:
        by_day.setdefault(item["date"], []).append(item)
    days_out = []
    all_flags = []
    for date in sorted(by_day):
        day_items = by_day[date]
        day_flags = _day_flags(day_items)
        day_flags.sort(key=lambda f: (_SEVERITY_RANK[f["severity"]],
                                      f["kind"], f["detail"]))
        all_flags.extend(day_flags)
        days_out.append({
            "date": date,
            "days_until": (datetime.date.fromisoformat(date) - today).days,
            "items": day_items,
            "flags": day_flags,
        })
    all_flags.sort(key=lambda f: (_SEVERITY_RANK[f["severity"]],
                                  f["date"], f["kind"], f["detail"]))
    counts = {
        "days": len(days_out),
        "items": len(items),
        "interviews": sum(1 for i in items if i["kind"] == "interview"),
        "offer_deadlines": sum(1 for i in items
                               if i["kind"] == "offer_deadline"),
        "posting_deadlines": sum(1 for i in items
                                 if i["kind"] == "posting_deadline"),
        "conflicts": sum(1 for f in all_flags if f["kind"] == "conflict"),
        "tight": sum(1 for f in all_flags if f["kind"] == "tight"),
        "busy": sum(1 for f in all_flags if f["kind"] == "busy"),
        "time_not_set": sum(1 for f in all_flags
                            if f["kind"] == "time-not-set"),
    }
    return {
        "days": days_out,
        "flags": all_flags,
        "counts": counts,
        "window": {"from": today.isoformat(), "to": end.isoformat(),
                   "days": days},
    }


def schedule_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_item(item):
    when = (f"{item['start']}–{item['end']}"
            if item["start"] else "all day")
    line = f"  {when}  {item['detail']}"
    if item["duration_assumed"]:
        line += "  [duration assumed: 1 hour]"
    return line


def _when_label(day):
    if day["days_until"] == 0:
        return "today"
    if day["days_until"] == 1:
        return "tomorrow"
    return f"in {day['days_until']}d"


def render_schedule_markdown(data):
    """Human-readable schedule: one day at a time, clashes first."""
    counts = data["counts"]
    window = data["window"]
    lines = [f"Schedule — {window['from']} to {window['to']} "
             f"({window['days']} days): {counts['days']} day(s) with "
             f"something on, {counts['items']} item(s) "
             f"({counts['interviews']} interview(s), "
             f"{counts['offer_deadlines']} offer deadline(s), "
             f"{counts['posting_deadlines']} closing date(s))"]
    flag_bits = []
    if counts["conflicts"]:
        flag_bits.append(f"{counts['conflicts']} clash(es)")
    if counts["tight"]:
        flag_bits.append(f"{counts['tight']} tight gap(s)")
    if counts["busy"]:
        flag_bits.append(f"{counts['busy']} busy day(s)")
    if counts["time_not_set"]:
        flag_bits.append(f"{counts['time_not_set']} round(s) with no "
                         "start time")
    if flag_bits:
        lines[0] += " — " + ", ".join(flag_bits)
    if not data["days"]:
        lines.append("Nothing scheduled in this window — no interviews, "
                     "offer deadlines, or closing dates in the next "
                     f"{window['days']} days. Use `interview --record` "
                     "to put a round on the calendar.")
        return "\n".join(lines) + "\n"
    lines.append("A clash is two interviews whose times overlap; a tight "
                 "gap is less than 30 minutes between two rounds on the "
                 "same day. All-day items are deadlines, not meetings — "
                 "two of them on one day is a busy day, not a clash.")
    for day in data["days"]:
        lines.append("")
        lines.append(f"{day['date']} ({_when_label(day)}) — "
                     f"{len(day['items'])} item(s)")
        for flag in day["flags"]:
            lines.append(f"  [{flag['severity']}] {flag['kind']}: "
                         f"{flag['detail']}")
        for item in day["items"]:
            lines.append(_fmt_item(item))
    return "\n".join(lines) + "\n"
