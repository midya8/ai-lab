"""Weekly application goals: a target to hit, and the streak behind it.

``autojob goals`` answers the only question a weekly target is for:
did I apply enough this week, and have I been keeping it up? A target
is set with ``goals --set-weekly N`` and stored in a small JSON file
next to the application log (``goals.json`` by default); the log itself
is never touched by this command — a target is a setting, not an event,
so it appears in no history and resets no clock.

The honest rules, carried over from the other modules:

- A week is Monday to Sunday, and an application counts in the week of
  its ``applied_at`` date. The current week is always shown as in
  progress: hitting the target early in the week still counts as met,
  and the week is not judged as missed until it ends.
- The streak counts consecutive weeks that met the target, walking back
  from the current week when it already met the target, otherwise from
  the last completed week. Weeks before your first logged application
  do not count against the streak — the streak starts when you started
  applying — and in the weekly table those weeks carry no verdict at
  all, instead of being scored as misses for a goal that did not exist
  yet. Exactly the target is met, not missed.
- With no target set there is nothing to meet and no streak to count:
  the command shows the weekly counts only and says so, instead of
  inventing a verdict. ``--set-weekly 0`` clears the target.
- An entry whose ``applied_at`` will not parse — a hand-edited log
  line — is skipped and counted, never fatal, so one bad line cannot
  hide the rest of the log.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

DEFAULT_WEEKS = 8


class GoalsError(ValueError):
    """A weekly target or goals file that cannot be used as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def default_goals_path(log_path):
    """The goals file that belongs with a log: ``goals.json`` beside it."""
    return str(Path(log_path).parent / "goals.json")


def load_goals(path):
    """Read the goals file. A missing file means no target set yet."""
    path = Path(path)
    if not path.exists():
        return {"weekly_target": None}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GoalsError(f"cannot read goals file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GoalsError(f"goals file {path} must hold one JSON object")
    target = data.get("weekly_target")
    if target is not None and (not isinstance(target, int)
                               or isinstance(target, bool) or target < 1):
        raise GoalsError(
            f"goals file {path}: weekly_target must be a whole number "
            f"of 1 or more, got {target!r}")
    return {"weekly_target": target}


def save_goals(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return data


def set_weekly_target(path, target):
    """Store a weekly target (``0`` clears it) and return the stored data."""
    if not isinstance(target, int) or isinstance(target, bool):
        raise GoalsError(
            f"--set-weekly must be a whole number, got {target!r}")
    if target < 0:
        raise GoalsError(f"--set-weekly must be 0 or more, got {target}")
    data = {"weekly_target": target if target > 0 else None}
    return save_goals(path, data)


def week_start(day):
    """The Monday of the week containing ``day``."""
    return day - datetime.timedelta(days=day.weekday())


def _streaks(counts, first_week, today_week, target):
    """Current and best streaks of weeks meeting ``target``.

    ``counts`` maps each Monday to the applications filed that week.
    The current streak walks back from the current week when it already
    met the target, otherwise from the last completed week, and stops
    at the first week that did not meet it or at ``first_week`` — the
    week of the first logged application, before which no week is
    judged. The best streak is the longest run of met weeks anywhere
    from the first application week to today.
    """
    if target is None:
        return None, None
    met = {}
    week = first_week
    while week <= today_week:
        met[week] = counts.get(week, 0) >= target
        week += datetime.timedelta(days=7)
    best = 0
    run = 0
    for week in sorted(met):
        run = run + 1 if met[week] else 0
        best = max(best, run)
    if met.get(today_week):
        start = today_week
    else:
        start = today_week - datetime.timedelta(days=7)
    current = 0
    while start >= first_week and met.get(start):
        current += 1
        start -= datetime.timedelta(days=7)
    return current, best


def collect_goals(entries, today=None, weeks=DEFAULT_WEEKS,
                  weekly_target=None):
    """Weekly application counts, this week against the target, and streaks.

    Returns {"weekly_target": ..., "this_week": {...},
    "current_streak": ..., "best_streak": ..., "weeks": [...],
    "counts": {"skipped": ...}}. ``today`` is a ``datetime.date``
    (defaults to the real current date; pass an explicit date in
    tests). ``weeks`` is how many weeks back the table runs, counting
    the current week. A week row carries its Monday, the applications
    filed that week, whether it met the target (``None`` before the
    first application or when no target is set), and whether it is the
    current week. Entries whose ``applied_at`` will not parse are
    counted under ``skipped`` instead of being guessed at.
    """
    if today is None:
        today = datetime.date.today()
    if weeks is not None and weeks < 1:
        raise ValueError(f"--weeks must be 1 or more, got {weeks}")
    counts = {}
    skipped = 0
    total = 0
    first_applied = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        applied = _parse_date(entry.get("applied_at"))
        if applied is None:
            skipped += 1
            continue
        total += 1
        if first_applied is None or applied < first_applied:
            first_applied = applied
        monday = week_start(applied)
        counts[monday] = counts.get(monday, 0) + 1
    today_week = week_start(today)
    first_week = week_start(first_applied) if first_applied is not None else None
    this_count = counts.get(today_week, 0)
    if weekly_target is None:
        this_met = None
        remaining = None
    else:
        this_met = this_count >= weekly_target
        remaining = max(0, weekly_target - this_count)
    if first_week is None:
        current_streak = best_streak = None
    else:
        current_streak, best_streak = _streaks(counts, first_week,
                                               today_week, weekly_target)
    rows = []
    monday = today_week
    for _ in range(weeks):
        count = counts.get(monday, 0)
        if weekly_target is None or first_week is None or monday < first_week:
            met = None
        else:
            met = count >= weekly_target
        rows.append({
            "week_start": monday.isoformat(),
            "week_end": (monday + datetime.timedelta(days=6)).isoformat(),
            "applications": count,
            "target": weekly_target,
            "met": met,
            "current": monday == today_week,
        })
        monday -= datetime.timedelta(days=7)
    return {
        "weekly_target": weekly_target,
        "this_week": {
            "week_start": today_week.isoformat(),
            "week_end": (today_week + datetime.timedelta(days=6)).isoformat(),
            "applications": this_count,
            "target": weekly_target,
            "met": this_met,
            "remaining": remaining,
        },
        "current_streak": current_streak,
        "best_streak": best_streak,
        "weeks": rows,
        "counts": {"total": total, "skipped": skipped},
    }


def goals_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_week(row):
    if row["applications"] == 1:
        applied = "1 application"
    else:
        applied = f"{row['applications']} applications"
    if row["target"] is None or row["met"] is None:
        # No target set, or the week predates the first logged
        # application: no verdict at all, not a miss.
        verdict = ""
    elif row["met"]:
        verdict = "  [met]"
    elif row["current"]:
        # The week is still in progress: the header already says how
        # many are left to go, so the row is not judged a miss yet.
        verdict = ""
    else:
        verdict = f"  [missed — {row['target'] - row['applications']} short]"
    current = "  [this week]" if row["current"] else ""
    return (f"{row['week_start']} to {row['week_end']}:  {applied}"
            f"{verdict}{current}")


def render_goals_markdown(data):
    """Human-readable weekly-goals view: this week, streaks, the table."""
    target = data["weekly_target"]
    this = data["this_week"]
    if target is None:
        header = ("Weekly goals — no target set yet; use "
                  "`goals --set-weekly N` to set one")
    else:
        header = (f"Weekly goals — target {target} application"
                  + ("s" if target != 1 else "") + " a week")
    lines = [header]
    if data["counts"]["total"] == 0:
        lines.append("No applications logged yet — apply to something "
                     "first, then set a weekly target with "
                     "`goals --set-weekly N`.")
        return "\n".join(lines) + "\n"
    if target is None:
        lines.append(f"This week ({this['week_start']} to "
                     f"{this['week_end']}): {this['applications']} "
                     f"application"
                     + ("s" if this["applications"] != 1 else "")
                     + " so far")
    elif this["met"]:
        lines.append(f"This week ({this['week_start']} to "
                     f"{this['week_end']}): {this['applications']} of "
                     f"{target} — target met")
    else:
        lines.append(f"This week ({this['week_start']} to "
                     f"{this['week_end']}): {this['applications']} of "
                     f"{target} — {this['remaining']} to go")
    if target is not None:
        if data["current_streak"]:
            lines.append(f"Current streak: {data['current_streak']} week"
                         + ("s" if data["current_streak"] != 1 else "")
                         + " meeting the target")
        else:
            lines.append("Current streak: 0 weeks — the last completed "
                         "week missed the target")
        lines.append(f"Best streak: {data['best_streak']} week"
                     + ("s" if data["best_streak"] != 1 else ""))
    lines.append("")
    for row in data["weeks"]:
        lines.append(f"  {_fmt_week(row)}")
    if data["counts"]["skipped"]:
        lines.append("")
        lines.append(f"({data['counts']['skipped']} logged application"
                     + ("s" if data["counts"]["skipped"] != 1 else "")
                     + " skipped: applied date would not parse)")
    return "\n".join(lines) + "\n"
