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
                    referrer=None, resume=None,
                    required_matched=None, required_missing=None,
                    preferred_matched=None, preferred_missing=None):
    """Append an application. Refuses duplicates on (company, title).

    The four skill lists (v0.6) record what the posting asked for and
    which of those the profile matched, so the digest can count gaps
    across applications. They are stored only when supplied — entries
    logged by older callers simply carry no skill data. ``deadline``
    (v0.15) is the posting's closing date as supplied by the caller;
    it is stored as a field on the entry, like ``match_score`` — a
    property of the posting, not an event. ``source`` (v0.18) is how
    the posting was found, one of the fixed channels in sources.py,
    and ``referrer`` is the name of the person who referred you (or
    the recruiter who brought the role); both are stored as fields on
    the entry, like ``deadline``. ``resume`` (v0.27) is the (already
    validated) ``{"path": ..., "note": ...}`` dict from
    ``resume.validate_resume`` — the resume file sent with the
    application — stored as the entry's ``resume`` field, like
    ``deadline`` and ``source``.
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
    if referrer is not None:
        entry["referrer"] = referrer
    if resume is not None:
        entry["resume"] = dict(resume)
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


def record_debrief(log_path, app_id, debrief, today=None):
    """Record a debrief (a write-up of one interview round).

    Appends the (already validated) debrief dict to the entry's
    ``debriefs`` list and a ``{"kind": "debrief", "at": date,
    "round": label}`` event to its history. A debrief is a write-up
    of one round, so re-recording the same (date, round) replaces the
    earlier write-up — and its history event — instead of adding a
    second rating for the same round. Refuses terminal applications:
    there is no round left to write up once an application is hired,
    rejected, or withdrawn. Recording a debrief does not move the
    application's status — that stays explicit through
    ``set_status``, the same rule as offers, interviews, and
    contacts. ``today`` defaults to the real current date; pass an
    explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record a debrief for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            _ensure_history(entry)
            debriefs = entry.get("debriefs")
            if not isinstance(debriefs, list):
                debriefs = []
                entry["debriefs"] = debriefs
            key = (debrief.get("date"), debrief.get("round"))
            debriefs[:] = [d for d in debriefs
                           if not (isinstance(d, dict)
                                   and (d.get("date"), d.get("round"))
                                   == key)]
            debriefs.append(debrief)
            entry["history"][:] = [
                event for event in entry["history"]
                if not (isinstance(event, dict)
                        and event.get("kind") == "debrief"
                        and (event.get("at"), event.get("round")) == key)]
            entry["history"].append({
                "kind": "debrief",
                "at": debrief["date"],
                "round": debrief.get("round"),
            })
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_note(log_path, app_id, note, today=None):
    """Record a dated note on an application.

    Appends the (already validated) note dict to the entry's
    ``notes`` list and a ``{"kind": "note", "at": date, "text":
    text, "tag": tag}`` event to its history. The event carries the
    text (and tag) so the timeline can render the note even when the
    record is missing from a hand-edited log. Multiple notes per
    application are allowed — a note is never deduplicated, and
    re-recording the same text on the same day is a second note, not
    an overwrite. Refuses terminal applications: there is nothing
    left to note once an application is closed. Recording a note
    does not move the application's status — that stays explicit
    through ``set_status``, the same rule as offers, interviews, and
    contacts. A note counts as activity, like a contact or a
    debrief: the history event resets the follow-up and stale clocks
    (see followup.last_activity), and it appears in the timeline
    feed; it never splits a stage stay in the analytics, which
    measure stays between status changes only. ``today`` defaults
    to the real current date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record a note for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            _ensure_history(entry)
            notes = entry.get("notes")
            if not isinstance(notes, list):
                notes = []
                entry["notes"] = notes
            notes.append(note)
            entry["history"].append({
                "kind": "note",
                "at": note["date"],
                "text": note.get("text"),
                "tag": note.get("tag"),
            })
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_onboard_task(log_path, app_id, task, today=None):
    """Record an onboarding task on a hired application.

    Appends the (already validated) task dict to the entry's
    ``onboarding`` list and a ``{"kind": "onboarding", "at": today,
    "task": text, "number": n, "done": False}`` event to its
    history. The task number is one more than the highest number
    already on the application, so numbers are never reused and a
    number you wrote down still means the same task. Refuses
    anything not hired: a live application has no onboarding to
    track yet, and a rejected or withdrawn one never will. A task
    is an event, like a contact or a debrief, so it appears in the
    timeline feed; it carries no status, so it never splits a stage
    stay in the analytics. ``today`` defaults to the real current
    date; pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") != "hired":
                raise TrackerError(
                    f"cannot record an onboarding task for {app_id}: "
                    f"status is {entry.get('status')!r} (not hired) — "
                    f"move it with `status --id {app_id} --to hired` first"
                )
            _ensure_history(entry)
            tasks = entry.get("onboarding")
            if not isinstance(tasks, list):
                tasks = []
                entry["onboarding"] = tasks
            numbers = [t.get("number") for t in tasks
                       if isinstance(t, dict)
                       and isinstance(t.get("number"), int)
                       and not isinstance(t.get("number"), bool)]
            number = (max(numbers) + 1) if numbers else 1
            record = {"number": number, "task": task["task"],
                      "category": task["category"], "due": task["due"],
                      "added_at": today, "done": False, "done_at": None}
            tasks.append(record)
            entry["history"].append({
                "kind": "onboarding",
                "at": today,
                "task": task["task"],
                "number": number,
                "done": False,
            })
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def complete_onboard_task(log_path, app_id, number, today=None):
    """Mark one onboarding task done.

    Sets ``done`` and ``done_at`` on the task with that number and
    appends a ``{"kind": "onboarding", "at": today, "task": text,
    "number": n, "done": True}`` event to the history, so the
    timeline shows both the task being written down and it being
    finished. Refuses a task that is already done, an unknown
    number, and anything not hired — the same rule as recording.
    ``today`` defaults to the real current date; pass an explicit
    date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") != "hired":
                raise TrackerError(
                    f"cannot complete an onboarding task for {app_id}: "
                    f"status is {entry.get('status')!r} (not hired)"
                )
            tasks = entry.get("onboarding") or []
            for record in tasks:
                if not isinstance(record, dict):
                    continue
                if record.get("number") != number:
                    continue
                if record.get("done"):
                    raise TrackerError(
                        f"onboarding task {number} for {app_id} is "
                        "already done")
                record["done"] = True
                record["done_at"] = today
                _ensure_history(entry)
                entry["history"].append({
                    "kind": "onboarding",
                    "at": today,
                    "task": record.get("task"),
                    "number": number,
                    "done": True,
                })
                _write_log(log_path, entries)
                return entry
            raise TrackerError(
                f"no onboarding task {number} on {app_id}")
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


def record_source(log_path, app_id, source):
    """Set (or replace) how an application was found.

    Stores the (already validated) ``{"source": ..., "referrer": ...}``
    dict from ``sources.validate_source`` as the entry's ``source``
    and ``referrer`` fields. Re-recording replaces both: a posting
    you thought came from a job board may turn out to have come from
    a recruiter, and a stale referrer must not linger when the source
    changes — so a record with no referrer removes the old one.
    Deliberately no history event: a source is a property of the
    application, not something that happened to it, so recording one
    must not reset the follow-up or stale clocks (see
    followup.last_activity, which takes the latest date of any
    history event) and does not appear in the timeline feed. Refuses
    terminal applications: there is nothing left to submit for them.
    """
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record a source for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            entry["source"] = source["source"]
            if source.get("referrer"):
                entry["referrer"] = source["referrer"]
            else:
                entry.pop("referrer", None)
            _write_log(log_path, entries)
            return entry
    raise TrackerError(f"no application with id {app_id!r}")


def record_resume(log_path, app_id, resume):
    """Set (or replace) which resume an application was sent with.

    Stores the (already validated) ``{"path": ..., "note": ...}``
    dict from ``resume.validate_resume`` as the entry's ``resume``
    field. Re-recording replaces it: you may learn later that the
    version you thought you sent was the older one, so this is a
    set, not an append. Deliberately no history event: which file
    you sent is a property of the application, not something that
    happened to it, so recording one must not reset the follow-up
    or stale clocks (see followup.last_activity, which takes the
    latest date of any history event) and does not appear in the
    timeline feed. Refuses terminal applications: there is nothing
    left to submit for them.
    """
    entries = read_log(log_path)
    for entry in entries:
        if entry["id"] == app_id:
            if entry.get("status") in TERMINAL:
                raise TrackerError(
                    f"cannot record a resume for {app_id}: status is "
                    f"{entry['status']!r} (terminal)"
                )
            entry["resume"] = dict(resume)
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
