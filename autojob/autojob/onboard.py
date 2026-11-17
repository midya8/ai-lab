"""Onboarding checklist: the first weeks at the new job, tracked.

``autojob onboard`` picks up where the rest of autojob stops. The
tracker ends at ``hired`` — the application is closed, the search is
over — but the first weeks at the new job still need doing: the
paperwork, the laptop, the people to meet, the things to learn
before you can do the job properly. ``onboard`` is a checklist for
that, on the hired application, with a due date per task so the
things that matter first do not slip.

The honest rules, carried over from the other modules:

- A task is a fact you supply: the words, a category, and a due
  date. The due date may be in the future — that is the whole point
  of a deadline — and a past date is allowed and flagged overdue,
  not refused, the same rule as posting closing dates: a task can
  already be late when you write it down.
- The category is one of six fixed words — paperwork, equipment,
  people, learning, logistics, other — so the list can be rolled up
  by kind of work. A task with no category is simply "other", never
  given a category you did not choose.
- Recording and completing a task only ever happens on a hired
  application. A live application has no onboarding to track yet,
  and a rejected or withdrawn one never will — recording one there
  is refused, with the status move that fixes it. Completing a task
  twice is refused too: a task is done once, and the record says
  when.
- A task is an event, like a contact or a debrief: it is stored as
  a history event (see ``tracker.record_onboard_task``), so it
  appears in the timeline feed. It never splits a stage stay in the
  analytics: stays are measured between status changes only, and a
  hired application has no stage left to split.
- Tasks are numbered per application, starting at 1, and the number
  never changes: completing task 2 does not renumber task 3, so a
  number you wrote down somewhere still means the same task. A
  task with no due date is listed, not hidden — it simply has no
  date for ``next`` to be urgent about, so it shows up there as an
  open item instead.
"""

from __future__ import annotations

import datetime
import json

ONBOARD_CATEGORIES = ("paperwork", "equipment", "people", "learning",
                      "logistics", "other")

DUE_SOON_DAYS = 7


class OnboardError(ValueError):
    """An onboarding task that cannot be recorded as stated."""


def validate_onboard_task(data, today=None):
    """Validate an onboarding task dict and return the cleaned form.

    Every problem is reported at once, like profile, offer,
    interview, contact, and debrief validation. ``today`` defaults
    to the real current date; pass an explicit date in tests. The
    returned dict carries the task, category, and due date; the
    tracker adds the number, the added date, and the done flag.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    errors = []
    data = data or {}

    task = data.get("task")
    if task is None or (isinstance(task, str) and not task.strip()):
        errors.append("task is required (the thing to do)")
    elif not isinstance(task, str):
        errors.append(f"task must be text, got {task!r}")

    category = data.get("category")
    cleaned_category = "other"
    if category is not None:
        if not isinstance(category, str):
            errors.append(f"category must be text, got {category!r}")
        elif not category.strip():
            errors.append("category cannot be blank — leave it out for "
                          "'other'")
        elif category.strip().lower() not in ONBOARD_CATEGORIES:
            errors.append(
                f"category must be one of {', '.join(ONBOARD_CATEGORIES)}, "
                f"got {category.strip()!r}")
        else:
            cleaned_category = category.strip().lower()

    due = data.get("due")
    cleaned_due = None
    if due is not None:
        if not isinstance(due, str):
            errors.append(f"due must be YYYY-MM-DD, got {due!r}")
        elif not due.strip():
            errors.append("due cannot be blank — leave it out for no "
                          "due date")
        else:
            try:
                datetime.date.fromisoformat(due.strip())
            except ValueError:
                errors.append(f"due must be YYYY-MM-DD, got {due!r}")
            else:
                cleaned_due = due.strip()

    if errors:
        raise OnboardError("; ".join(errors))

    return {
        "task": task.strip(),
        "category": cleaned_category,
        "due": cleaned_due,
    }


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _clean_task(task):
    """A stored task as a list row, or None when it is not one.

    A task that is not a dict, or carries no task text (a
    hand-edited log), is skipped, never fatal. A task with no
    parseable due date is kept — it simply has no date to be urgent
    about — and a task with no number sorts by its recorded order.
    """
    if not isinstance(task, dict):
        return None
    text = task.get("task")
    if not isinstance(text, str) or not text.strip():
        return None
    category = task.get("category")
    if not isinstance(category, str) or not category.strip():
        category = "other"
    else:
        category = category.strip().lower()
    if category not in ONBOARD_CATEGORIES:
        category = "other"
    number = task.get("number")
    if not (isinstance(number, int) and not isinstance(number, bool)):
        number = None
    due = task.get("due")
    if not isinstance(due, str) or not due.strip():
        due = None
    else:
        due = due.strip()
    return {"number": number, "task": text.strip(), "category": category,
            "due": due, "done": bool(task.get("done")),
            "done_at": task.get("done_at")
            if isinstance(task.get("done_at"), str) else None,
            "added_at": task.get("added_at")
            if isinstance(task.get("added_at"), str) else None,
            "when": _parse_date(due)}


def collect_onboard(entries, today=None, app_id=None):
    """Every onboarding task on a hired application.

    Returns {"tasks": [...], "summary": {...}}. Each row carries the
    application id, company, title, task number, task, category, due
    date, days left (negative when overdue, None with no due date),
    state (overdue / today / soon / later / no-date / done), and the
    done date. Pending tasks come first — overdue first, then due
    soonest, then no due date — and done tasks last, most recently
    done first. Only hired applications are listed: a live
    application has no onboarding yet, and a rejected or withdrawn
    one never will. ``app_id`` shows one application; the summary
    counts the tasks, the hired applications, and the hired
    applications with nothing recorded. ``today`` is a
    ``datetime.date`` (defaults to the real current date; pass an
    explicit date in tests).
    """
    if today is None:
        today = datetime.date.today()
    rows = []
    by_category = {}
    hired = 0
    without_tasks = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "hired":
            continue
        hired += 1
        tasks = entry.get("onboarding") or []
        cleaned = [row for row in (_clean_task(t) for t in tasks)
                   if row is not None]
        if not cleaned:
            without_tasks += 1
        # The roll-up counts the whole log, not the filter, the same
        # rule as the notes tag roll-up.
        for task in cleaned:
            by_category[task["category"]] = (
                by_category.get(task["category"], 0) + 1)
        if app_id is not None and entry.get("id") != app_id:
            continue
        for task in cleaned:
            days_left = None
            state = "no-date"
            if task["done"]:
                state = "done"
            elif task["when"] is not None:
                days_left = (task["when"] - today).days
                if days_left < 0:
                    state = "overdue"
                elif days_left == 0:
                    state = "today"
                elif days_left <= DUE_SOON_DAYS:
                    state = "soon"
                else:
                    state = "later"
            rows.append({
                "id": entry.get("id"),
                "company": entry.get("company"),
                "title": entry.get("title"),
                "number": task["number"],
                "task": task["task"],
                "category": task["category"],
                "due": task["due"],
                "days_left": days_left,
                "state": state,
                "done": task["done"],
                "done_at": task["done_at"],
                "added_at": task["added_at"],
            })

    def _pending_key(row):
        # Overdue first (most overdue first), then due soonest, then
        # no due date; within one due date the lower number first.
        due = row["days_left"]
        return (0 if due is not None and due < 0 else
                1 if due is not None else 2,
                due if due is not None else 10 ** 9,
                row["number"] if row["number"] is not None else 10 ** 9)

    pending = sorted((r for r in rows if not r["done"]),
                     key=_pending_key)
    done = sorted((r for r in rows if r["done"]),
                  key=lambda r: (r["done_at"] or "",
                                 r["number"] if r["number"] is not None
                                 else 10 ** 9),
                  reverse=True)
    ordered = pending + done
    summary = {
        "total": len(rows),
        "done": len(done),
        "pending": len(pending),
        "overdue": sum(1 for r in pending if r["state"] == "overdue"),
        "due_soon": sum(1 for r in pending if r["state"] in ("today", "soon")),
        "hired_applications": hired,
        "without_tasks": without_tasks,
        "by_category": dict(sorted(by_category.items(),
                                   key=lambda pair: (-pair[1], pair[0]))),
    }
    return {"tasks": ordered, "summary": summary}


def onboard_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    number = f"#{row['number']}" if row["number"] is not None else "#?"
    head = f"{number}  {row['task']} [{row['category']}]"
    if row["done"]:
        head += f" — done {row['done_at'] or 'date not recorded'}"
        if row["due"]:
            head += f" (was due {row['due']})"
    elif row["due"]:
        if row["days_left"] is not None and row["days_left"] < 0:
            head += (f" — due {row['due']}, "
                     f"overdue by {-row['days_left']}d")
        elif row["days_left"] == 0:
            head += f" — due {row['due']}, today"
        elif row["days_left"] == 1:
            head += f" — due {row['due']}, tomorrow"
        elif row["days_left"] is not None:
            head += f" — due {row['due']}, in {row['days_left']}d"
        else:
            head += f" — due {row['due']}"
    else:
        head += " — no due date"
    return (f"{head} — {row['title']} at {row['company']} "
            f"({row['id']})")


def render_onboard_markdown(data):
    """Human-readable onboarding list: pending first, done last."""
    tasks = data["tasks"]
    summary = data["summary"]
    header = (f"Onboarding — {summary['total']} task"
              + ("s" if summary["total"] != 1 else "")
              + f" ({summary['done']} done, {summary['pending']} pending")
    if summary["overdue"]:
        header += f", {summary['overdue']} overdue"
    header += (f") across {summary['hired_applications']} hired "
               f"application"
               + ("s" if summary["hired_applications"] != 1 else ""))
    lines = [header]
    if summary["by_category"]:
        lines.append("  " + ", ".join(
            f"{count} {category}" for category, count
            in summary["by_category"].items()))
    if summary["without_tasks"]:
        lines.append(
            f"  {summary['without_tasks']} hired application"
            + ("s" if summary["without_tasks"] != 1 else "")
            + " with no onboarding tasks recorded yet")
    if not tasks:
        lines.append("No onboarding tasks recorded yet — use "
                     "`onboard --record --id APP-001 --task "
                     "'Complete I-9 paperwork' --due YYYY-MM-DD` to "
                     "start one, on an application that reached hired.")
        return "\n".join(lines) + "\n"
    lines.append("")
    pending = [row for row in tasks if not row["done"]]
    done = [row for row in tasks if row["done"]]
    for row in pending:
        lines.append(f"  {_fmt_row(row)}")
    if done:
        lines.append("")
        lines.append("  Done:")
        for row in done:
            lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
