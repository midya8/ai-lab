"""Interview debriefs: write up a round while it is still fresh.

``debrief --record`` stores a write-up of one interview round on a
tracker entry — the date and round label it belongs to, a rating from
1 to 5, what went well, what to improve, and the questions they
actually asked. ``debrief`` on its own lists the write-ups, newest
first, names the past rounds that still have no write-up, and rolls
up the questions that keep coming up across interviews.

The honest rules, carried over from interview.py and contacts.py:

- Everything recorded is a fact the user supplies. The rating, the
  went-well and to-improve notes, and the questions are all the user's
  own words — this module never generates text about an interview, so
  it cannot invent a question, an answer, or an outcome. The
  interviewer shown on a debrief row comes from the interview round
  it pairs with, exactly as recorded there.
- A debrief is a write-up of one round, so re-recording the same
  (date, round) replaces the earlier write-up instead of adding a
  second one — two ratings for one round would make the average
  meaningless. A round is a date, and the date cannot be in the
  future: a debrief is written after the round happened, not before.
- A debrief is not a status: recording one does not move the
  application's status — that stays explicit through ``status``, the
  same rule as offers, interviews, and contacts. Recording is refused
  on a terminal application (hired / rejected / withdrawn): there is
  no round left to write up once an application is closed.
- A debrief counts as activity, like a contact: it is stored as a
  history event, so writing one up resets the follow-up and stale
  clocks, and it appears in the timeline feed. It never splits a
  stage stay in the analytics — stays are measured between status
  changes only.
- A debrief may name a round that was never recorded as an interview
  — a call you never scheduled in the tool still deserves a write-up
  — and the list says so (``matched: false``) instead of pretending
  the round is on file. The "rounds without a debrief" list only ever
  names rounds that *were* recorded, so it cannot nag about a round
  the tool never knew about.
- A round dated today counts as ready to write up. Interviews treat
  today as upcoming, because the round may still be ahead of you; a
  debrief is written after the fact, and a round earlier today is
  already over.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES


class DebriefError(ValueError):
    """A debrief that cannot be recorded as stated."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate_debrief(data, today=None):
    """Validate a debrief dict and return the cleaned form.

    Every problem is reported at once, like profile, offer, and
    interview validation. ``today`` defaults to the real current date;
    pass an explicit date in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    errors = []
    data = data or {}

    date = data.get("date")
    when = None
    if date is None or date == "":
        errors.append("date is required (the date of the round)")
    else:
        try:
            when = datetime.date.fromisoformat(date)
        except (ValueError, TypeError):
            errors.append(f"date must be YYYY-MM-DD, got {date!r}")
    if when is not None:
        try:
            ref = datetime.date.fromisoformat(today)
        except (ValueError, TypeError):
            ref = None
        if ref is not None and when > ref:
            errors.append("date cannot be in the future — a debrief is "
                          "a write-up of a round that already happened")

    round_label = data.get("round")
    if round_label is None or (isinstance(round_label, str)
                               and not round_label.strip()):
        errors.append("round is required (e.g. 'Hiring manager')")
    elif not isinstance(round_label, str):
        errors.append(f"round must be text, got {round_label!r}")

    rating = data.get("rating")
    if rating is not None:
        if not _is_int(rating) or rating < 1 or rating > 5:
            errors.append(f"rating must be a whole number from 1 to 5, "
                          f"got {rating!r}")

    went_well = data.get("went_well")
    if went_well is not None and not isinstance(went_well, str):
        errors.append(f"went_well must be text, got {went_well!r}")

    improve = data.get("improve")
    if improve is not None and not isinstance(improve, str):
        errors.append(f"improve must be text, got {improve!r}")

    questions = data.get("questions")
    cleaned_questions = None
    if questions is not None:
        if not isinstance(questions, list):
            errors.append(f"questions must be a list, got {questions!r}")
        else:
            cleaned_questions = []
            for question in questions:
                if not isinstance(question, str) or not question.strip():
                    errors.append("each question must be non-empty text, "
                                  f"got {question!r}")
                    break
                cleaned_questions.append(question.strip())

    if errors:
        raise DebriefError("; ".join(errors))

    return {
        "date": date,
        "round": round_label.strip(),
        "rating": rating,
        "went_well": went_well.strip() if isinstance(went_well, str)
                     and went_well.strip() else None,
        "improve": improve.strip() if isinstance(improve, str)
                   and improve.strip() else None,
        "questions": cleaned_questions if cleaned_questions is not None
                     else [],
        "recorded_at": today,
    }


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _interview_by_key(entry):
    """{(date, round): interview} for one entry, first record wins."""
    by_key = {}
    for interview in entry.get("interviews") or []:
        if not isinstance(interview, dict):
            continue
        key = (interview.get("date"), interview.get("round"))
        if key not in by_key:
            by_key[key] = interview
    return by_key


def collect_debriefs(entries, today=None):
    """Every recorded debrief, newest first, plus the gaps.

    Returns {"debriefs": [...], "missing": [...], "summary": {...}}.
    ``missing`` is every past round (dated today or earlier) on a live
    application with no write-up yet, most recent first. ``summary``
    carries the counts, the average rating over rated debriefs, how
    many past rounds have a write-up, and the questions asked more
    than once across interviews. ``today`` is a ``datetime.date``
    (defaults to the real current date; pass an explicit date in
    tests). Debriefs and rounds on terminal applications are left
    out, the same rule as the contacts list.
    """
    if today is None:
        today = datetime.date.today()
    debriefs = []
    missing = []
    debriefed_keys = set()
    past_rounds = 0
    questions_seen = {}
    rated = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") in TERMINAL_STATUSES:
            continue
        interviews = _interview_by_key(entry)
        for debrief in entry.get("debriefs") or []:
            if not isinstance(debrief, dict):
                continue
            when = _parse_date(debrief.get("date"))
            if when is None:
                continue
            key = (entry.get("id"), debrief.get("date"),
                   debrief.get("round"))
            if key in debriefed_keys:
                continue
            debriefed_keys.add(key)
            interview = interviews.get((debrief.get("date"),
                                        debrief.get("round")))
            rating = debrief.get("rating")
            if _is_int(rating):
                rated.append(rating)
            questions = debrief.get("questions") or []
            if isinstance(questions, list):
                for question in set(
                        q.strip().lower() for q in questions
                        if isinstance(q, str) and q.strip()):
                    seen = questions_seen.setdefault(
                        question, {"display": None, "asked": 0,
                                   "companies": set()})
                    if seen["display"] is None:
                        seen["display"] = next(
                            q.strip() for q in questions
                            if isinstance(q, str)
                            and q.strip().lower() == question)
                    seen["asked"] += 1
                    if entry.get("company"):
                        seen["companies"].add(entry["company"])
            debriefs.append({
                "id": entry.get("id"),
                "company": entry.get("company"),
                "title": entry.get("title"),
                "status": entry.get("status"),
                "date": debrief["date"],
                "round": debrief.get("round"),
                "interviewer": (interview or {}).get("interviewer"),
                "rating": rating if _is_int(rating) else None,
                "went_well": debrief.get("went_well"),
                "improve": debrief.get("improve"),
                "questions": ([q for q in questions
                               if isinstance(q, str)]
                              if isinstance(questions, list) else []),
                "days_ago": (today - when).days,
                "matched": interview is not None,
            })
        for (when_str, round_label), interview in interviews.items():
            when = _parse_date(when_str)
            if when is None or when > today:
                continue
            past_rounds += 1
            if (entry.get("id"), when_str, round_label) in debriefed_keys:
                continue
            missing.append({
                "id": entry.get("id"),
                "company": entry.get("company"),
                "title": entry.get("title"),
                "status": entry.get("status"),
                "date": when_str,
                "round": round_label,
                "interviewer": interview.get("interviewer"),
                "days_ago": (today - when).days,
            })

    debriefs.sort(key=lambda row: (row["date"],
                                   (row["company"] or "").lower(),
                                   (row["title"] or "").lower(),
                                   (row["round"] or "").lower()),
                  reverse=True)
    missing.sort(key=lambda row: (row["date"],
                                  (row["company"] or "").lower(),
                                  (row["title"] or "").lower(),
                                  (row["round"] or "").lower()),
                 reverse=True)
    repeated = [{"question": seen["display"], "asked": seen["asked"],
                 "companies": sorted(seen["companies"])}
                for seen in questions_seen.values() if seen["asked"] > 1]
    repeated.sort(key=lambda item: (-item["asked"],
                                    item["question"].lower()))
    return {
        "debriefs": debriefs,
        "missing": missing,
        "summary": {
            "total": len(debriefs),
            "rated": len(rated),
            "average_rating": (round(sum(rated) / len(rated), 1)
                               if rated else None),
            "rounds_with_debrief": past_rounds - len(missing),
            "rounds_total": past_rounds,
            "questions_total": sum(len(row["questions"])
                                   for row in debriefs),
            "repeated_questions": repeated,
        },
    }


def debriefs_for(entries, app_id, today=None):
    """All debriefs (and missing rounds) for one application.

    Returns the entry plus its debrief rows and missing rounds, or
    (None, [], []) when the id is unknown. The entry is returned even
    when it holds no debriefs, so the caller can tell "known
    application, nothing written up" apart from "no such
    application". Terminal applications return their entry with empty
    lists, the same rule as the full list.
    """
    if today is None:
        today = datetime.date.today()
    for entry in entries:
        if entry.get("id") == app_id:
            data = collect_debriefs([entry], today=today)
            return entry, data["debriefs"], data["missing"]
    return None, [], []


def debrief_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    parts = [f"{row['date']}  {row['title']} — {row['company']}  "
             f"{row['round']}"]
    if row.get("interviewer"):
        parts.append(f"with {row['interviewer']}")
    if row.get("rating") is not None:
        parts.append(f"rated {row['rating']}/5")
    if row["days_ago"] == 0:
        parts.append("— today")
    elif row["days_ago"] == 1:
        parts.append("— yesterday")
    elif row["days_ago"] > 1:
        parts.append(f"— {row['days_ago']}d ago")
    else:
        parts.append(f"— in {-row['days_ago']}d")
    line = " ".join(parts)
    if not row.get("matched"):
        line += "  [no matching interview recorded]"
    if row.get("went_well"):
        line += f"\n      went well: {row['went_well']}"
    if row.get("improve"):
        line += f"\n      to improve: {row['improve']}"
    if row.get("questions"):
        line += f"\n      asked ({len(row['questions'])}): " + "; ".join(
            row["questions"])
    return line


def _fmt_missing(row):
    parts = [f"{row['date']}  {row['title']} — {row['company']}  "
             f"{row['round']}"]
    if row.get("interviewer"):
        parts.append(f"with {row['interviewer']}")
    if row["days_ago"] == 0:
        parts.append("— today")
    elif row["days_ago"] == 1:
        parts.append("— yesterday")
    else:
        parts.append(f"— {row['days_ago']}d ago")
    return " ".join(parts)


def render_debrief_markdown(data):
    """Human-readable debrief list: newest first, then the gaps."""
    debriefs = data["debriefs"]
    missing = data["missing"]
    summary = data["summary"]
    header = (f"Debriefs — {summary['total']} recorded")
    if summary["rated"]:
        header += (f", average rating {summary['average_rating']}/5 "
                   f"({summary['rated']} rated)")
    header += (f", {summary['rounds_with_debrief']} of "
               f"{summary['rounds_total']} past rounds written up")
    lines = [header]
    if not debriefs and not missing:
        lines.append("No debriefs recorded yet — use "
                     "`debrief --record --id APP-001 --date YYYY-MM-DD "
                     "--round 'Hiring manager' --rating 4` to write one up.")
        return "\n".join(lines) + "\n"
    if debriefs:
        lines.append("")
        for row in debriefs:
            lines.append(f"  {_fmt_row(row)}")
    if missing:
        lines.append("")
        lines.append("Rounds without a debrief yet:")
        for row in missing:
            lines.append(f"  {_fmt_missing(row)}")
    repeated = summary["repeated_questions"]
    if repeated:
        lines.append("")
        lines.append("Questions asked more than once:")
        for item in repeated:
            companies = ", ".join(item["companies"])
            lines.append(f"  {item['asked']}x — {item['question']}"
                         + (f" ({companies})" if companies else ""))
    return "\n".join(lines) + "\n"
