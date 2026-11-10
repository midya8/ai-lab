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
                    packet_dir=None, today=None):
    """Append an application. Refuses duplicates on (company, title)."""
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
