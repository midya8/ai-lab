"""Next actions: one prioritised list of what to do today.

``autojob next`` pulls together everything the other commands already
know — follow-ups due, interviews coming up, posting closing dates,
offer decision deadlines, contacts gone quiet, applications with no
movement, and rejected applications still missing a reason — and
orders them by urgency, so the first line is the thing to do first.
It records nothing and changes nothing; it only reads the log.

The honest rules, carried over from the other modules:

- Every action names the application it belongs to and the date it is
  anchored to, so nothing in the list is a guess. A dated action
  (interview, closing date, offer deadline) is anchored to that date;
  an overdue one (follow-up, quiet contact, stale application) is
  anchored to the last recorded activity — the day it went quiet.
- Order is urgency first, then date: overdue items before today's,
  today's before soon, soon before later, and a rejected application
  with no reason recorded — which has no date to be urgent about —
  last. Within the same urgency and date, interviews come first
  (they are fixed in someone else's calendar), then closing dates,
  then offer decisions, then follow-ups, then contacts, then stale
  applications, then recording a rejection reason.
- The window is 7 days by default (``--days N`` changes it). Overdue
  actions are always shown, whatever the window: an overdue item does
  not stop being urgent because it fell outside the window. A dated
  action past the window is left out, not shown as later.
- Terminal applications (hired / rejected / withdrawn) never produce
  a follow-up, interview, deadline, offer, contact, or stale action:
  there is nothing left to do for them. The one exception is a
  rejected application with no reason recorded, which produces a
  single action to record the reason — that is bookkeeping, not
  chasing.
- A follow-up and a stale flag for the same application are two
  different actions, not a duplicate: one is a nudge to send, the
  other is a prompt to check whether the application is still live.
  ``--kind`` narrows the list when you only want one sort of work.
- A date that will not parse — a hand-edited log line — is skipped,
  never fatal, in every source the list draws on.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES, stale_applications
from .contacts import collect_contacts
from .deadlines import collect_deadlines
from .followup import due_followups
from .interview import collect_interviews
from .offer import latest_offer
from .rejections import NOT_RECORDED, collect_rejections

NEXT_KINDS = (
    "followup",
    "interview",
    "deadline",
    "offer",
    "contact",
    "stale",
    "rejection-reason",
)

# Order within the same urgency and date: fixed appointments first,
# then dates that close, then nudges, then bookkeeping.
KIND_ORDER = {
    "interview": 0,
    "deadline": 1,
    "offer": 2,
    "followup": 3,
    "contact": 4,
    "stale": 5,
    "rejection-reason": 6,
}

URGENCY_ORDER = {
    "overdue": 0,
    "today": 1,
    "soon": 2,
    "later": 3,
    "open": 4,
}

DEFAULT_DAYS = 7
SOON_DAYS = 7


class NextError(ValueError):
    """A next-actions list that cannot be built as asked."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _urgency_for(days_left):
    """overdue / today / soon (1-7 days) / later, from days to a date."""
    if days_left < 0:
        return "overdue"
    if days_left == 0:
        return "today"
    if days_left <= SOON_DAYS:
        return "soon"
    return "later"


def _when_phrase(urgency, date, today):
    """How far away the anchor date is, in plain words."""
    if date is None:
        return "no date recorded"
    days = (date - today).days
    if urgency == "overdue":
        return f"overdue by {-days}d" if days < 0 else "due"
    if urgency == "today":
        return "today"
    if days == 1:
        return "tomorrow"
    return f"in {days}d"


def _action(kind, urgency, date, entry, summary, detail):
    return {
        "kind": kind,
        "urgency": urgency,
        "date": date.isoformat() if date else None,
        "id": entry.get("id"),
        "company": entry.get("company"),
        "title": entry.get("title"),
        "status": entry.get("status"),
        "summary": summary,
        "detail": detail,
    }


def _followup_actions(entries, today, wait_days):
    actions = []
    for item in due_followups(entries, today, wait_days):
        entry = {"id": item["id"], "company": item["company"],
                 "title": item["title"], "status": item["status"]}
        last = _parse_date(item["last_activity"])
        sent = item["followups_sent"]
        detail = (f"last activity {item['last_activity']}, "
                  f"{item['days_waiting']} days waiting"
                  + (f" ({sent} already sent)" if sent
                     else " (none sent yet)"))
        actions.append(_action(
            "followup", "overdue", last, entry,
            item["suggestion"].capitalize(), detail))
    return actions


def _interview_actions(entries, today, days):
    actions = []
    data = collect_interviews(entries, today=today)
    for row in data["upcoming"]:
        if row["status"] in TERMINAL_STATUSES:
            continue
        if row["days_until"] > days:
            continue
        entry = {"id": row["id"], "company": row["company"],
                 "title": row["title"], "status": row["status"]}
        when = _parse_date(row["date"])
        summary = f"Interview: {row['round'] or 'round'}"
        if row.get("interviewer"):
            summary += f" with {row['interviewer']}"
        parts = [part for part in
                 (row.get("mode"),
                  (f"{row['duration_minutes']} min"
                   if row.get("duration_minutes") else None),
                  (f"at {row['time']}" if row.get("time") else None))
                 if part]
        detail = ", ".join(parts) if parts else "no details recorded"
        actions.append(_action(
            "interview", _urgency_for(row["days_until"]), when, entry,
            summary, detail))
    return actions


def _deadline_actions(entries, today, days):
    actions = []
    data = collect_deadlines(entries, today=today, days=days)
    for row in data["deadlines"]:
        entry = {"id": row["id"], "company": row["company"],
                 "title": row["title"], "status": row["status"]}
        when = _parse_date(row["deadline"])
        days_left = row["days_left"]
        if days_left < 0:
            summary = f"Posting closed {-days_left}d ago"
        elif days_left == 0:
            summary = "Posting closes today"
        elif days_left == 1:
            summary = "Posting closes tomorrow"
        else:
            summary = f"Posting closes in {days_left}d"
        actions.append(_action(
            "deadline", row["state"], when, entry, summary,
            f"closing date {row['deadline']}"))
    return actions


def _offer_actions(entries, today, days):
    actions = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        offer = latest_offer(entry)
        if offer is None:
            continue
        deadline = _parse_date(offer.get("deadline"))
        if deadline is None:
            continue
        days_left = (deadline - today).days
        if days_left > days:
            continue
        if days_left < 0:
            summary = f"Offer decision was due {-days_left}d ago"
        elif days_left == 0:
            summary = "Offer decision due today"
        elif days_left == 1:
            summary = "Offer decision due tomorrow"
        else:
            summary = f"Offer decision due in {days_left}d"
        base = offer.get("base_annual")
        currency = offer.get("currency")
        if isinstance(base, (int, float)) and currency:
            detail = f"base {currency} {base:,.0f} a year"
        else:
            detail = "no figures recorded"
        detail += f"; deadline {offer['deadline']}"
        actions.append(_action(
            "offer", _urgency_for(days_left), deadline, entry,
            summary, detail))
    return actions


def _contact_actions(entries, today, quiet_days):
    actions = []
    data = collect_contacts(entries, today=today, quiet_days=quiet_days)
    for row in data["contacts"]:
        if not row["quiet"]:
            continue
        entry = {"id": row["id"], "company": row["company"],
                 "title": row["title"], "status": row["status"]}
        last = _parse_date(row["last_contact"])
        summary = f"Check in with {row['name']}"
        if row.get("role"):
            summary += f" ({row['role']})"
        detail = f"last spoke {row['days_since_touch']}d ago"
        if row.get("channel"):
            detail += f" via {row['channel']}"
        actions.append(_action(
            "contact", "overdue", last, entry, summary, detail))
    return actions


def _stale_actions(entries, today, stale_days):
    actions = []
    for item in stale_applications(entries, stale_days, today):
        entry = {"id": item["id"], "company": item["company"],
                 "title": item["title"], "status": item["status"]}
        last = today - datetime.timedelta(days=item["days_since_last_move"])
        actions.append(_action(
            "stale", "overdue", last, entry,
            "No movement — check whether this is still live",
            f"no movement in {item['days_since_last_move']}d"))
    return actions


def _rejection_reason_actions(entries, today):
    actions = []
    data = collect_rejections(entries, today=today)
    for row in data["rejections"]:
        if row["reason"] != NOT_RECORDED:
            continue
        entry = {"id": row["id"], "company": row["company"],
                 "title": row["title"], "status": "rejected"}
        when = _parse_date(row["rejected_at"])
        detail = "no reason recorded yet"
        if row.get("stage_reached"):
            detail += f"; reached {row['stage_reached']}"
        if row.get("rejected_at"):
            detail += f"; rejected {row['rejected_at']}"
        actions.append(_action(
            "rejection-reason", "open", when, entry,
            "Record why this rejection happened", detail))
    return actions


_BUILDERS = {
    "followup": lambda entries, today, opts: _followup_actions(
        entries, today, opts["wait_days"]),
    "interview": lambda entries, today, opts: _interview_actions(
        entries, today, opts["days"]),
    "deadline": lambda entries, today, opts: _deadline_actions(
        entries, today, opts["days"]),
    "offer": lambda entries, today, opts: _offer_actions(
        entries, today, opts["days"]),
    "contact": lambda entries, today, opts: _contact_actions(
        entries, today, opts["quiet_days"]),
    "stale": lambda entries, today, opts: _stale_actions(
        entries, today, opts["stale_days"]),
    "rejection-reason": lambda entries, today, opts: (
        _rejection_reason_actions(entries, today)),
}


def collect_next(entries, today=None, days=DEFAULT_DAYS, kinds=None,
                 wait_days=None, quiet_days=14, stale_days=14):
    """Every action due across the log, most urgent first.

    Returns {"actions": [...], "counts": {...}, "window_days": days,
    "today": iso}. Each action carries its kind, urgency, anchor date,
    application identity, and a summary plus the detail behind it.
    ``today`` is a ``datetime.date`` (defaults to the real current
    date; pass an explicit date in tests). ``days`` is how far ahead
    dated actions are looked for — overdue ones are always included.
    ``kinds`` narrows the list to those kinds of work. ``wait_days``,
    ``quiet_days`` and ``stale_days`` are the bars the follow-up,
    contact, and stale checks use, with the same defaults and the
    same boundaries as the commands they come from.
    """
    if today is None:
        today = datetime.date.today()
    if days is None or days < 0:
        raise NextError(f"--days must be 0 or more, got {days}")
    for name, value in (("wait-days", wait_days),
                        ("quiet-days", quiet_days),
                        ("stale-days", stale_days)):
        if value is not None and value < 0:
            raise NextError(f"--{name} must be 0 or more, got {value}")
    wanted = list(NEXT_KINDS) if kinds is None else list(kinds)
    unknown = [kind for kind in wanted if kind not in NEXT_KINDS]
    if unknown:
        raise NextError(
            f"kind must be one of {', '.join(NEXT_KINDS)}, "
            f"got {', '.join(unknown)}")
    opts = {"days": days, "wait_days": wait_days,
            "quiet_days": quiet_days, "stale_days": stale_days}
    actions = []
    for kind in wanted:
        actions.extend(_BUILDERS[kind](entries, today, opts))

    def _sort_key(action):
        when = _parse_date(action["date"])
        ordinal = when.toordinal() if when is not None else 10 ** 9
        return (URGENCY_ORDER[action["urgency"]], ordinal,
                KIND_ORDER[action["kind"]],
                (action["company"] or "").lower(),
                (action["title"] or "").lower())

    actions.sort(key=_sort_key)
    counts = {"total": len(actions)}
    for urgency in URGENCY_ORDER:
        counts[urgency] = sum(1 for action in actions
                              if action["urgency"] == urgency)
    counts["by_kind"] = {kind: sum(1 for action in actions
                                   if action["kind"] == kind)
                         for kind in NEXT_KINDS}
    return {"actions": actions, "counts": counts,
            "window_days": days, "today": today.isoformat()}


def next_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_action(action, today):
    when = _parse_date(action["date"])
    date_text = action["date"] or "—"
    line = (f"{date_text}  [{action['urgency']}]  {action['kind']}  "
            f"{action['id']}  {action['company']} — {action['title']}  — "
            f"{action['summary']} ({action['detail']}; "
            f"{_when_phrase(action['urgency'], when, today)})")
    return line


def render_next_markdown(data):
    """Human-readable next-actions list: most urgent first."""
    counts = data["counts"]
    today = _parse_date(data["today"]) or datetime.date.today()
    header = (f"Next actions — {counts['total']} "
              f"(next {data['window_days']} day"
              + ("s" if data["window_days"] != 1 else "") + ")")
    parts = []
    for urgency in ("overdue", "today", "soon", "later", "open"):
        if counts[urgency]:
            parts.append(f"{counts[urgency]} {urgency}")
    if parts:
        header += "; " + ", ".join(parts)
    lines = [header]
    by_kind = ", ".join(
        f"{kind}: {counts['by_kind'][kind]}" for kind in NEXT_KINDS)
    lines.append(f"  {by_kind}")
    if not data["actions"]:
        lines.append("")
        lines.append("Nothing due — no follow-ups due, no interviews, "
                     "closing dates or offer deadlines in the window, "
                     "no quiet contacts, no stale applications, and "
                     "every rejection has a reason recorded.")
        return "\n".join(lines) + "\n"
    lines.append("")
    for action in data["actions"]:
        lines.append(f"  {_fmt_action(action, today)}")
    return "\n".join(lines) + "\n"
