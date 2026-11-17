"""Resume versions: which resume file each application actually got.

You iterate on a resume — a sharper summary, a reworded bullet, a
version tuned for one posting — and after a few weeks you cannot
remember which file went where. ``autojob resume`` records it: the
resume file sent with each application, so when a reply comes back
you know what they read.

The honest rules, carried over from the other modules:

- A resume is a label you supply: the file name or path of the
  resume you sent (``resume-data-analyst-v3.pdf``), plus an optional
  note ("the one with the shorter summary"). autojob never opens the
  file and never checks it exists — the resume lives wherever you
  keep it, and a path recorded on another machine is still a true
  record. What is stored is exactly what you typed, apart from
  surrounding whitespace.
- A resume is a field on the tracker entry (see
  ``tracker.record_resume``), not a history event: which file you
  sent is a property of the application, not something that happened
  to it, so recording one does not reset the follow-up or stale
  clocks, and it does not appear in the timeline feed.
  Re-recording replaces it — you may learn later that the version
  you thought you sent was the older one — so this is a set, not an
  append. Recording a resume is refused on a terminal application,
  the same rule as a closing date or a source: the resume belongs on
  the application while it is live.
- ``apply --resume PATH`` records the resume at apply time, before
  anything is written — a bad value exits 1 with no packet and no
  log line. ``batch --resume PATH`` sends the same resume with every
  application in the batch. Applications logged before v0.27 carry
  no resume; they are counted as missing, not guessed.
- The list shows one row per application with the resume it got (or
  "no resume recorded"), and a roll-up of how many applications
  each version went to — so you can see at a glance whether the new
  version has actually been sent anywhere yet. ``--missing`` shows
  only the applications with nothing recorded; ``--resume PATH``
  shows only the ones that got that version.
"""

from __future__ import annotations

import datetime
import json


class ResumeError(ValueError):
    """A resume that cannot be recorded as stated."""


def validate_resume(path, note=None):
    """Validate a resume record and return the cleaned form.

    Every problem is reported at once, like profile, offer,
    interview, contact, and rejection validation. The path is
    required text (surrounding whitespace is stripped; the stored
    value is exactly what you typed otherwise — autojob never opens
    the file). The note is optional free text, kept verbatim apart
    from surrounding whitespace.
    """
    errors = []
    cleaned_path = None
    if path is None or (isinstance(path, str) and not path.strip()):
        errors.append("resume is required (the file sent with the "
                      "application, e.g. resume-data-analyst-v3.pdf)")
    elif not isinstance(path, str):
        errors.append(f"resume must be text, got {path!r}")
    else:
        cleaned_path = path.strip()
    cleaned_note = None
    if note is not None:
        if not isinstance(note, str):
            errors.append(f"note must be text, got {note!r}")
        elif note.strip():
            cleaned_note = note.strip()
    if errors:
        raise ResumeError("; ".join(errors))
    return {"path": cleaned_path, "note": cleaned_note}


def recorded_resume(entry):
    """The entry's recorded resume, or None.

    A stored resume that is not a dict with a non-empty text path
    (a hand-edited log, or an entry logged before v0.27) is treated
    as unrecorded, never fatal.
    """
    resume = entry.get("resume")
    if not isinstance(resume, dict):
        return None
    path = resume.get("path")
    if not isinstance(path, str) or not path.strip():
        return None
    note = resume.get("note")
    if not isinstance(note, str) or not note.strip():
        note = None
    else:
        note = note.strip()
    return {"path": path.strip(), "note": note}


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def collect_resumes(entries, today=None, resume=None, missing_only=False,
                    app_id=None):
    """One row per application, with the resume it got.

    Returns {"rows": [...], "counts": {...}}. Each row carries the
    application id, company, title, status, applied date, days since
    applying, and the recorded resume (path + note, or None). Rows
    are newest applied first; entries with no parseable ``applied_at``
    sort last. ``counts`` rolls up how many applications each resume
    version went to (most-used first), how many carry no resume, and
    the total — over the whole log, not just the filtered rows.
    ``resume`` narrows the rows to applications that got that exact
    version (surrounding whitespace ignored); ``missing_only`` keeps
    only applications with nothing recorded; ``app_id`` shows one
    application. Entries that are not dicts are skipped, never
    fatal.
    """
    if today is None:
        today = datetime.date.today()
    by_resume = {}
    no_resume = 0
    total = 0
    rows = []
    wanted = resume.strip() if isinstance(resume, str) else None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        total += 1
        recorded = recorded_resume(entry)
        if recorded is None:
            no_resume += 1
        else:
            by_resume[recorded["path"]] = by_resume.get(recorded["path"], 0) + 1
        if app_id is not None and entry.get("id") != app_id:
            continue
        if missing_only and recorded is not None:
            continue
        if wanted is not None and (recorded is None
                                   or recorded["path"] != wanted):
            continue
        applied = _parse_date(entry.get("applied_at"))
        rows.append({
            "id": entry.get("id"),
            "company": entry.get("company"),
            "title": entry.get("title"),
            "status": entry.get("status"),
            "applied_at": entry.get("applied_at"),
            "days_since_applied": ((today - applied).days
                                   if applied is not None else None),
            "resume": recorded,
        })
    rows.sort(key=lambda row: row.get("id") or "")
    rows.sort(key=lambda row: (row["applied_at"] is not None
                               and _parse_date(row["applied_at"]) is not None,
                               _parse_date(row["applied_at"])
                               or datetime.date.min),
              reverse=True)
    return {
        "rows": rows,
        "counts": {
            "total": total,
            "no_resume": no_resume,
            "with_resume": total - no_resume,
            "by_resume": dict(sorted(by_resume.items(),
                                     key=lambda pair: (-pair[1], pair[0]))),
        },
    }


def resumes_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    head = f"{row['id']} — {row['company']}, {row['title']}"
    details = []
    if row["status"]:
        details.append(row["status"])
    if row["applied_at"]:
        details.append(f"applied {row['applied_at']}")
    if row["days_since_applied"] is not None:
        days = row["days_since_applied"]
        details.append(f"{days}d ago" if days != 1 else "1d ago")
    if details:
        head += f" ({'; '.join(details)})"
    resume = row["resume"]
    if resume is None:
        return head + " — no resume recorded"
    text = head + f" — {resume['path']}"
    if resume["note"]:
        text += f" ({resume['note']})"
    return text


def render_resumes_markdown(data):
    """Human-readable resume list: versions first, then one row per
    application."""
    rows = data["rows"]
    counts = data["counts"]
    header = (f"Resumes — {counts['total']} application"
              + ("s" if counts["total"] != 1 else ""))
    if counts["no_resume"]:
        header += (f"; {counts['no_resume']} with no resume recorded")
    lines = [header]
    if counts["by_resume"]:
        lines.append("")
        lines.append("  Versions:")
        for path, count in counts["by_resume"].items():
            lines.append(f"    {path}: {count} application"
                         + ("s" if count != 1 else ""))
    if not rows:
        lines.append("No applications match — record which resume went "
                     "with one using `resume --record --id APP-001 "
                     "--file resume-data-analyst-v3.pdf`, or pass "
                     "`--resume` to `apply`.")
        return "\n".join(lines) + "\n"
    lines.append("")
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
