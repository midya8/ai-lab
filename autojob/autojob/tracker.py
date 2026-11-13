"""Application tracker: a JSONL log of applications, with dedupe and status.

One line per application. Statuses move forward through a fixed state
machine; terminal states (rejected / withdrawn / hired) have no outgoing
transitions. Applying to the same company + title twice is refused — use
``status`` to update the existing entry instead.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

STATUSES = ("applied", "screening", "interview", "offer", "hired", "rejected", "withdrawn")
TERMINAL = ("hired", "rejected", "withdrawn")
ALLOWED = {
    "applied": ("screening", "interview", "rejected", "withdrawn"),
    "screening": ("interview", "rejected", "withdrawn"),
    "interview": ("offer", "rejected", "withdrawn"),
    "offer": ("hired", "rejected", "withdrawn"),
    "hired": (),
    "rejected": (),
    "withdrawn": (),
}


class TrackerError(ValueError):
    """Dedupe, unknown id, or illegal status transition."""


def _norm(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def slug(text):
    """URL/file-safe slug: lowercase, non-alphanumerics become hyphens."""
    out = re.sub(r"[^a-z0-9]+", "-", (text or "").strip().lower()).strip("-")
    return out or "unknown"


def read_log(path):
    """Read the JSONL log. Missing file means no applications yet."""
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TrackerError(f"{path}:{i}: corrupt log line: {exc}") from exc
    return entries


def _write_log(path, entries):
    text = "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries)
    Path(path).write_text(text, encoding="utf-8")


def next_id(entries):
    return f"APP-{len(entries) + 1:03d}"


def add_application(log_path, company, title, source=None, match_score=None,
                    packet_dir=None, today=None, deadline=None,
                    required_matched=None, required_missing=None,
                    preferred_matched=None, preferred_missing=None):
    """Append an application. Refuses duplicates on (company, title).

    The four skill lists (v0.6) record what the posting asked for and
    which of those the profile matched, so the digest can count gaps
    across applications. They are stored only when supplied — entries
    logged by older callers simply carry no skill data. ``deadline``
    (v0.15) is the posting's closing date as supplied by the caller;
    it is stored as a field on the entry, like ``match_score`` — a
    property of the posting, not an event.
    """
    entries = read_log(log_path)
    for entry in entries:
        if _norm(entry["company"]) == _norm(company) and _norm(entry["title"]) == _norm(title):
            raise TrackerError(
                f"already applied to {company!r} / {title!r} as {entry['id']} "
                f"(status: {entry['status']}); use `status` to update it"
            )
    if today is None:
        today = datetime.date.today().isoformat()
    entry = {
        "id": next_id(entries),
        "company": company,
        "title": title,
        "source": source,
        "status": "applied",
        "applied_at": today,
        "history": [{"status": "applied", "at": today}],
        "match_score": match_score,
        "packet_dir": packet_dir,
    }
    if deadline is not None:
        entry["deadline"] = deadline
    if required_matched is not None:
        entry["required_matched"] = list(required_matched)
    if required_missing is not None:
        entry["required_missing"] = list(required_missing)
    if preferred_matched is not None:
        entry["preferred_matched"] = list(preferred_matched)
    if preferred_missing is not None:
        entry["preferred_missing"] = list(preferred_missing)
    entries.append(entry)
    _write_log(log_path, entries)
    return entry


def _ensure_history(entry):
    """Give a pre-v0.2 entry a history it can grow from.

    The entry is known to have reached its current status at some unknown
    date, so the current stage is backfilled at ``applied_at`` — the
    earliest it could have happened. Stage durations computed from such
    backfilled entries are upper bounds, not measured stays; see
    analytics.days_in_stage, which only counts stays between two recorded
    events.
    """
    history = entry.get("history")
    if isinstance(history, list) and history:
        return history
    history = []
    if entry.get("applied_at"):
        history.append({"status": "applied", "at": entry["applied_at"]})
    current = entry.get("status")
    if current and current != "applied" and entry.get("applied_at"):
        history.append({"status": current, "at": entry["applied_at"]})
    entry["history"] = history
    return history


def set_status(log_path, app_id, new_status, today=None):
    """Move an application to a new status, enforcing the state machine.

    Records the move in the entry's history (``today`` defaults to the
    real current date; pass an explicit date in tests). A move to the
    status the entry already has is a no-op and records nothing.
    """
    if new_status not in STATUSES:
        raise TrackerError(f"unknown status {new_status!r}; one of: {', '.join(STATUSES)}")
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            current = entry["status"]
            if new_status == current:
                return entry
            if new_status not in ALLOWED[current]:
                raise TrackerError(
                    f"cannot move {app_id} from {current!r} to {new_status!r}; "
                    f"allowed from {current}: {', '.join(ALLOWED[current]) or 'none (terminal)'}"
                )
            # Backfill first: _ensure_history reads the *old* status.
            history = _ensure_history(entry)
            entry["status"] = new_status
            history.append({"status": new_status, "at": today})
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_offer(log_path, app_id, offer, today=None):
    """Record an offer on an application.

    Appends the (already validated) offer dict to the entry's ``offers``
    list and a ``{"kind": "offer", "at": today}`` event to its history.
    Multiple offers per application are allowed — a renegotiated offer
    is a new record, not an overwrite; the comparison uses the latest.
    Refuses terminal applications: there is no offer to record once an
    application is hired, rejected, or withdrawn. Recording an offer
    does not move the application's status — that stays explicit
    through ``set_status``. ``today`` defaults to the real current
    date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record an offer for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            _ensure_history(entry)
            offers = entry.get("offers")
            if not isinstance(offers, list):
                offers = []
                entry["offers"] = offers
            offers.append(offer)
            entry["history"].append({"kind": "offer", "at": today})
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_interview(log_path, app_id, interview, today=None):
    """Record an interview round on an application.

    Appends the (already validated) interview dict to the entry's
    ``interviews`` list and a ``{"kind": "interview", "at": date,
    "round": label}`` event to its history. Multiple rounds per
    application are allowed — including two on the same day — so a
    round is never deduplicated; a re-recorded round is a new record,
    not an overwrite. Refuses terminal applications: there is no
    interview to schedule once an application is hired, rejected, or
    withdrawn. Recording a round does not move the application's
    status — that stays explicit through ``set_status``, the same rule
    as offers. ``today`` defaults to the real current date; pass an
    explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record an interview for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            _ensure_history(entry)
            rounds = entry.get("interviews")
            if not isinstance(rounds, list):
                rounds = []
                entry["interviews"] = rounds
            rounds.append(interview)
            entry["history"].append({
                "kind": "interview",
                "at": interview["date"],
                "round": interview.get("round"),
            })
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_contact(log_path, app_id, contact, today=None):
    """Record a contact (a person) on an application.

    Appends the (already validated) contact dict to the entry's
    ``contacts`` list and a ``{"kind": "contact", "at": date,
    "name": name}`` event to its history. Recording the same person
    again is a new touch, not a duplicate — the list groups by name
    and takes the latest touch, so a re-recorded contact moves that
    person's last-spoke date forward. Refuses terminal applications:
    there is nobody left to talk to once an application is hired,
    rejected, or withdrawn. Recording a contact does not move the
    application's status — that stays explicit through ``set_status``,
    the same rule as offers and interviews. ``today`` defaults to the
    real current date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record a contact for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            _ensure_history(entry)
            contacts = entry.get("contacts")
            if not isinstance(contacts, list):
                contacts = []
                entry["contacts"] = contacts
            contacts.append(contact)
            entry["history"].append({
                "kind": "contact",
                "at": contact["date"],
                "name": contact.get("name"),
            })
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_deadline(log_path, app_id, deadline, today=None):
    """Set (or move) the posting closing date on an application.

    Stores the (already validated) date as the entry's ``deadline``
    field. Re-recording moves the date — a posting's closing date can
    change — so this is a set, not an append. Deliberately no history
    event: a closing date is a property of the posting, not something
    that happened to the application, so recording one must not reset
    the follow-up or stale clocks (see followup.last_activity, which
    takes the latest date of any history event) and does not appear in
    the timeline feed. Refuses terminal applications: there is nothing
    left to submit for them. ``today`` is accepted for symmetry with
    the other record functions but the closing date carries its own
    date, so it is ignored.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record a closing date for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            entry["deadline"] = deadline
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_rejection(log_path, app_id, rejection, today=None):
    """Record why a rejected application ended.

    Stores the (already validated) rejection dict — category plus an
    optional note — as the entry's ``rejection`` field, with the date
    it was recorded. Re-recording replaces the reason: you may learn
    the real one later, so this is a set, not an append. Deliberately
    no history event: the rejection itself is already in the history
    as a status move, so recording the reason must not reset the
    follow-up or stale clocks (see followup.last_activity, which takes
    the latest date of any history event) and does not appear in the
    timeline feed. Refuses anything that is not already ``rejected``:
    a live application has not been rejected yet, and a hired or
    withdrawn one was not rejected at all — move it to ``rejected``
    first with ``set_status``. ``today`` defaults to the real current
    date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") != "rejected":
                raise TrackerError(
                    f"cannot record a rejection reason for {app_id}: "
                    f"status is {entry['status']!r} (must be 'rejected')"
                )
            stored = dict(rejection)
            stored["recorded_at"] = today
            entry["rejection"] = stored
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def find_application(entries, app_id):
    for entry in entries:
        if entry["id"] == app_id:
            return entry
    return None


def record_followup(log_path, app_id, today=None, note=None):
    """Record that a follow-up was sent for an application.

    Appends a ``{"kind": "followup", "at": today}`` event to the entry's
    history, which resets the follow-up clock (see followup.due_followups)
    and the stale clock (see analytics.stale_applications). Refuses
    terminal applications — there is nobody left to follow up with — and
    refuses a second follow-up recorded on the same day, so a double
    invocation cannot fake two nudges. ``today`` defaults to the real
    current date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot follow up on {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            history = _ensure_history(entry)
            for event in history:
                if (event.get("kind") == "followup"
                        and event.get("at") == today):
                    raise TrackerError(
                        f"a follow-up for {app_id} was already recorded "
                        f"on {today}"
                    )
            event = {"kind": "followup", "at": today}
            if note:
                event["note"] = note
            history.append(event)
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")
