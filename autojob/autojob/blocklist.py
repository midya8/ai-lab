"""Blocklist: the companies you will not apply to again.

``autojob blocklist --record --company NAME --reason TEXT --date
YYYY-MM-DD`` writes a company down with the reason, in your own
words — ghosted after a final round, a salary band that will not
move, a "remote" listing that turned out to be hybrid. Once a
company is on the list, ``apply`` refuses it, and ``rank`` and
``batch`` skip its postings with the reason shown, so a good score
cannot talk you back into a place you already ruled out.

The honest rules, carried over from contacts and the question bank:

- A block is a setting, not an event. It lives in a small
  ``blocklist.json`` beside the application log (``--blocklist``
  points somewhere else), the same pattern as ``goals.json`` and
  ``questions.json``. Recording or removing one never touches the
  log, appears in no history, and resets no clock.
- The company and the date are facts you supply, never defaulted:
  the date is the day you decided, and a record without one is
  refused. The reason is your own words — this module checks that
  a company was named, never what the reason says.
- Matching is by company name only, case and extra spaces ignored,
  so "vector labs" and "Vector Labs" are one entry; re-blocking
  the same company replaces the earlier reason and date instead of
  adding a second row. Removing a company takes it off the list
  entirely — the decision is yours to reverse, and ``rank`` and
  ``batch`` treat it as never blocked.
- The list only gates future applications. It does not hide or
  change applications already in the log: if you blocked a company
  after applying, the logged application stays where it is, and
  ``next``, ``timeline``, and the other log commands still show it.
  What the list stops is the next packet, not the record of the
  last one.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path


class BlocklistError(ValueError):
    """A blocked company or blocklist file that cannot be used as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _normalize(company):
    """The grouping key for a company: words only, case ignored."""
    return " ".join(str(company).split()).casefold()


def default_blocklist_path(log_path):
    """The blocklist file that belongs with a log: ``blocklist.json``
    beside it."""
    return str(Path(log_path).parent / "blocklist.json")


def load_blocklist(path):
    """Read the blocklist file. A missing file means nothing blocked yet."""
    path = Path(path)
    if not path.exists():
        return {"blocked": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlocklistError(
            f"cannot read blocklist file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise BlocklistError(
            f"blocklist file {path} must hold one JSON object")
    blocked = data.get("blocked", [])
    if not isinstance(blocked, list):
        raise BlocklistError(
            f"blocklist file {path}: 'blocked' must hold a list")
    cleaned = []
    for record in blocked:
        if not isinstance(record, dict):
            raise BlocklistError(
                f"blocklist file {path}: each blocked company must hold "
                f"one JSON object")
        company = record.get("company")
        reason = record.get("reason")
        blocked_on = record.get("blocked_on")
        if not isinstance(company, str) or not company.strip():
            raise BlocklistError(
                f"blocklist file {path}: a blocked company has no name")
        if reason is not None and not isinstance(reason, str):
            raise BlocklistError(
                f"blocklist file {path}: reason for {company!r} must be "
                f"text")
        if not isinstance(blocked_on, str) or _parse_date(blocked_on) is None:
            raise BlocklistError(
                f"blocklist file {path}: {company!r} has no valid "
                f"blocked date")
        cleaned.append({
            "company": company.strip(),
            "reason": reason.strip() if reason and reason.strip() else None,
            "blocked_on": blocked_on.strip(),
        })
    return {"blocked": cleaned}


def save_blocklist(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return data


def validate_block(company, reason, date):
    """Validate a block and return the cleaned form.

    Every problem is reported at once, like contact and debrief
    validation. The reason is the user's own words — this function
    checks that one is not required, never what it says.
    """
    errors = []
    if company is None or (isinstance(company, str)
                           and not company.strip()):
        errors.append("company is required (the company not to apply to)")
    elif not isinstance(company, str):
        errors.append(f"company must be text, got {company!r}")
    if reason is not None and not isinstance(reason, str):
        errors.append(f"reason must be text, got {reason!r}")
    if date is None or (isinstance(date, str) and not date.strip()):
        errors.append("date is required (the day you decided, YYYY-MM-DD)")
    elif not isinstance(date, str) or _parse_date(date) is None:
        errors.append(f"date must be YYYY-MM-DD, got {date!r}")
    if errors:
        raise BlocklistError("; ".join(errors))
    return {
        "company": company.strip(),
        "reason": reason.strip() if reason and reason.strip() else None,
        "blocked_on": date.strip(),
    }


def add_block(path, company, reason, date):
    """Block a company, replacing any earlier block for the same
    company. Returns (record, replaced)."""
    cleaned = validate_block(company, reason, date)
    data = load_blocklist(path)
    key = _normalize(cleaned["company"])
    replaced = False
    for i, record in enumerate(data["blocked"]):
        if _normalize(record["company"]) == key:
            data["blocked"][i] = cleaned
            replaced = True
            break
    else:
        data["blocked"].append(cleaned)
    save_blocklist(path, data)
    return cleaned, replaced


def remove_block(path, company):
    """Take a company off the blocklist. Returns the removed record,
    or None when the company was not blocked."""
    data = load_blocklist(path)
    key = _normalize(company)
    removed = None
    kept = []
    for record in data["blocked"]:
        if _normalize(record["company"]) == key and removed is None:
            removed = record
        else:
            kept.append(record)
    if removed is None:
        return None
    save_blocklist(path, {"blocked": kept})
    return removed


def is_blocked(entries, company):
    """The block record for a company, or None when it is not blocked.

    ``entries`` is the list ``load_blocklist`` returns under
    "blocked" (or the dict itself). Matching ignores case and extra
    spaces, so a posting's "Vector Labs" matches a stored
    "vector labs".
    """
    if isinstance(entries, dict):
        entries = entries.get("blocked", [])
    if company is None:
        return None
    key = _normalize(company)
    for record in entries or []:
        if isinstance(record, dict) and _normalize(
                record.get("company")) == key:
            return record
    return None


def collect_blocklist(entries):
    """Every blocked company, one row each, sorted by company name.

    Returns {"blocked": [...], "count": n}. Each row carries the
    company as first written, the reason (or None), and the date it
    was blocked.
    """
    if isinstance(entries, dict):
        entries = entries.get("blocked", [])
    rows = [{
        "company": record["company"],
        "reason": record.get("reason"),
        "blocked_on": record.get("blocked_on"),
    } for record in entries or [] if isinstance(record, dict)]
    rows.sort(key=lambda row: row["company"].casefold())
    return {"blocked": rows, "count": len(rows)}


def blocklist_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    head = f"{row['company']} — "
    if row["reason"]:
        head += row["reason"]
    else:
        head += "no reason given"
    head += f" (blocked {row['blocked_on']})"
    return head


def render_blocklist_markdown(data):
    """Human-readable blocklist: companies you will not apply to."""
    rows = data["blocked"]
    company_word = ("company" if data["count"] == 1 else "companies")
    header = (f"Blocklist — {data['count']} {company_word} you will not "
              f"apply to")
    lines = [header]
    if not rows:
        lines.append("No companies blocked — record one with `blocklist "
                     "--record --company NAME --reason TEXT "
                     "--date YYYY-MM-DD`, and `apply`, `rank`, and "
                     "`batch` will refuse or skip it.")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
