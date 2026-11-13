"""Rejection analysis: why applications ended, and what they had in common.

``rejections --record`` stores the reason a rejected application ended —
a short category from a fixed list, plus a note in your own words.
``rejections`` on its own lists every rejected application with the stage
it reached, how long it took to hear back, and the skills those postings
asked for that the profile lacked.

The honest rules, carried over from the other modules:

- A rejection reason is a fact you supply, about an application that
  already ended. It is stored as a field on the tracker entry (see
  ``tracker.record_rejection``), not as a history event: the rejection
  itself is already in the history as a status move, so recording the
  reason does not reset the follow-up or stale clocks, and it does not
  appear in the timeline feed. Re-recording replaces the reason — you
  may learn the real one later — so this is a set, not an append.
- Recording a reason is refused unless the application is already
  ``rejected``. A live application has not been rejected yet, and a
  hired or withdrawn one was not rejected at all; the status stays
  explicit through ``status``, the same rule as offers, interviews,
  and contacts. If an application went quiet without a formal no,
  move it to ``rejected`` first and then record ``ghosted``.
- The reason is a category, not a diagnosis. ``skills-gap`` means the
  posting asked for a skill the profile lacks — it does not prove the
  skill is why you were turned down. The missing-skills roll-up counts
  how often each skill was missing on rejected applications; it is a
  pattern to look at, not a cause. The note carries what you were
  actually told, in your own words.
- The stage reached comes from the tracker's recorded history: the
  status immediately before the move to ``rejected``. An application
  that went applied → rejected reached ``applied``; one that reached
  the interview stage says so. Days to rejection run from
  ``applied_at`` to the recorded rejection date; both dates must parse,
  and a row with an unusable date shows no day count instead of a
  guessed one. Applications rejected before history existed (pre-v0.2
  entries) show no dates at all.
- Rejections with no reason recorded are counted, not hidden: the list
  says how many carry no reason, so an unrecorded reason is visible
  work, not a silent gap.
"""

from __future__ import annotations

import datetime
import json

REJECTION_REASONS = (
    "ghosted",
    "skills-gap",
    "experience-level",
    "salary",
    "location",
    "internal-hire",
    "role-closed",
    "culture-fit",
    "other",
)

NOT_RECORDED = "not recorded"


class RejectionError(ValueError):
    """A rejection reason that cannot be recorded as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def validate_rejection(category, note=None):
    """Validate a rejection reason and return the cleaned form.

    Every problem is reported at once, like profile, offer, interview,
    and contact validation. The category must be one of
    REJECTION_REASONS; the note is free text and is kept verbatim
    apart from surrounding whitespace.
    """
    errors = []
    if category is None or (isinstance(category, str) and not category.strip()):
        errors.append("reason is required")
    elif not isinstance(category, str):
        errors.append(f"reason must be one of {', '.join(REJECTION_REASONS)}, "
                      f"got {category!r}")
    elif category not in REJECTION_REASONS:
        errors.append(f"reason must be one of {', '.join(REJECTION_REASONS)}, "
                      f"got {category!r}")
    if note is not None and not isinstance(note, str):
        errors.append(f"note must be text, got {note!r}")
    if errors:
        raise RejectionError("; ".join(errors))
    return {
        "category": category,
        "note": note.strip() if isinstance(note, str) and note.strip() else None,
    }


def _recorded_reason(entry):
    """The entry's recorded rejection reason, or None.

    A hand-edited ``rejection`` field with an unknown category is
    treated as unrecorded, never fatal.
    """
    rejection = entry.get("rejection")
    if not isinstance(rejection, dict):
        return None
    if rejection.get("category") not in REJECTION_REASONS:
        return None
    return rejection


def _rejection_facts(entry):
    """(rejected_at, stage_reached) from the entry's status history.

    The rejection date is the date of the move to ``rejected``; the
    stage reached is the status immediately before that move. Either
    is None when the history does not record it — a hand-edited log,
    or an entry whose history was never written.
    """
    history = entry.get("history")
    if not isinstance(history, list):
        history = []
    status_events = [
        (event.get("status"), _parse_date(event.get("at")))
        for event in history
        if isinstance(event, dict) and "status" in event
    ]
    for i, (status, when) in enumerate(status_events):
        if status == "rejected":
            stage = status_events[i - 1][0] if i > 0 else None
            return when, stage
    return None, None


def _skill_list(entry, key):
    value = entry.get(key)
    if not isinstance(value, list):
        return []
    return [skill for skill in value if isinstance(skill, str) and skill]


def _counted(mapping):
    """A {skill: count} dict ordered by count desc, then name."""
    return dict(sorted(mapping.items(), key=lambda pair: (-pair[1], pair[0])))


def collect_rejections(entries, today=None, reason=None):
    """Every rejected application, most recently rejected first.

    Returns {"rejections": [...], "counts": {...},
    "median_days_to_reject": float|None,
    "missing": {"required": {skill: n}, "preferred": {skill: n}}}.
    Each row carries the entry's identity, the date it was rejected
    and the stage it reached (both None when the history does not
    record them), days from applying to rejection and days since the
    rejection, the recorded reason or "not recorded", the note, the
    match score, and the required/preferred skills the posting asked
    for that the profile lacked. ``today`` is a ``datetime.date``
    (defaults to the real current date; pass an explicit date in
    tests). ``reason`` narrows the rows to one recorded reason; an
    unknown reason raises ValueError.
    """
    if today is None:
        today = datetime.date.today()
    if reason is not None and reason not in REJECTION_REASONS:
        raise ValueError(
            f"reason must be one of {', '.join(REJECTION_REASONS)}, "
            f"got {reason!r}")
    rows = []
    missing_required = {}
    missing_preferred = {}
    days = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "rejected":
            continue
        rejected_at, stage = _rejection_facts(entry)
        applied_at = _parse_date(entry.get("applied_at"))
        if rejected_at is not None and applied_at is not None:
            days_to_reject = (rejected_at - applied_at).days
        else:
            days_to_reject = None
        days_since = ((today - rejected_at).days
                      if rejected_at is not None else None)
        recorded = _recorded_reason(entry)
        row_reason = recorded["category"] if recorded else NOT_RECORDED
        if reason is not None and row_reason != reason:
            continue
        if days_to_reject is not None:
            days.append(days_to_reject)
        required_missing = _skill_list(entry, "required_missing")
        preferred_missing = _skill_list(entry, "preferred_missing")
        for skill in required_missing:
            missing_required[skill] = missing_required.get(skill, 0) + 1
        for skill in preferred_missing:
            missing_preferred[skill] = missing_preferred.get(skill, 0) + 1
        rows.append({
            "id": entry.get("id"),
            "company": entry.get("company"),
            "title": entry.get("title"),
            "rejected_at": rejected_at.isoformat() if rejected_at else None,
            "stage_reached": stage,
            "days_to_reject": days_to_reject,
            "days_since_rejection": days_since,
            "match_score": entry.get("match_score"),
            "required_missing": required_missing,
            "preferred_missing": preferred_missing,
            "reason": row_reason,
            "note": recorded.get("note") if recorded else None,
        })
    # Most recently rejected first; rows with no date sort last, and
    # rows on the same date keep company/title order.
    def _sort_key(row):
        when = _parse_date(row["rejected_at"])
        if when is None:
            return (1, 0, (row["company"] or "").lower(),
                    (row["title"] or "").lower())
        return (0, -when.toordinal(), (row["company"] or "").lower(),
                (row["title"] or "").lower())
    rows.sort(key=_sort_key)
    by_reason = {}
    by_stage = {}
    for row in rows:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        stage = row["stage_reached"] or "unknown"
        by_stage[stage] = by_stage.get(stage, 0) + 1
    return {
        "rejections": rows,
        "counts": {
            "total": len(rows),
            "by_reason": _counted(by_reason),
            "by_stage": _counted(by_stage),
            "not_recorded": by_reason.get(NOT_RECORDED, 0),
        },
        "median_days_to_reject": _median(days),
        "missing": {
            "required": _counted(missing_required),
            "preferred": _counted(missing_preferred),
        },
    }


def rejections_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    parts = [f"{row['id']}  {row['company']} — {row['title']}"]
    stage = row["stage_reached"] or "unknown stage"
    parts.append(f"[reached {stage}]")
    if row["rejected_at"] is None:
        parts.append("rejected (date not recorded)")
    else:
        when = f"rejected {row['rejected_at']}"
        if row["days_to_reject"] is not None:
            when += f" ({row['days_to_reject']}d after applying"
            if row["days_since_rejection"] is not None:
                when += f", {row['days_since_rejection']}d ago"
            when += ")"
        parts.append(when)
    parts.append(f"— {row['reason']}")
    line = " ".join(parts)
    details = []
    if row.get("note"):
        details.append(f"note: {row['note']}")
    missing = []
    for skill in row["required_missing"]:
        missing.append(f"{skill} (required)")
    for skill in row["preferred_missing"]:
        missing.append(f"{skill} (preferred)")
    if missing:
        details.append("missing: " + ", ".join(missing))
    if details:
        line += "\n      " + "\n      ".join(details)
    return line


def _fmt_counts(mapping):
    return ", ".join(f"{name} {count}"
                     for name, count in mapping.items())


def render_rejections_markdown(data):
    """Human-readable rejection list: most recently rejected first."""
    rows = data["rejections"]
    counts = data["counts"]
    header = (f"Rejections — {counts['total']} rejected application"
              + ("s" if counts["total"] != 1 else ""))
    median = data["median_days_to_reject"]
    if median is not None:
        header += f", median {median:.1f} days to a decision"
    if counts["not_recorded"]:
        header += (f"; {counts['not_recorded']} with no reason recorded")
    lines = [header]
    if not rows:
        lines.append("No rejected applications yet — when one ends, move "
                     "it to `rejected` with `status`, then record why "
                     "with `rejections --record --id APP-001 "
                     "--reason ghosted`.")
        return "\n".join(lines) + "\n"
    lines.append(f"  by reason: {_fmt_counts(counts['by_reason'])}")
    lines.append(f"  furthest stage reached: {_fmt_counts(counts['by_stage'])}")
    missing = data["missing"]
    missing_parts = []
    for skill, n in missing["required"].items():
        missing_parts.append(f"{skill} (required, {n})")
    for skill, n in missing["preferred"].items():
        missing_parts.append(f"{skill} (preferred, {n})")
    if missing_parts:
        lines.append("  most often missing on rejected applications: "
                     + ", ".join(missing_parts))
    lines.append("")
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
