"""Interview question bank: the questions that keep coming up, and the
answers you have prepared for them.

``autojob questions`` reads the questions out of every debrief in the
log and groups them — same question, asked by different companies,
is one row — most-asked first, so the question that keeps coming up
is the one you prepare for next. ``questions --record --question
TEXT --answer TEXT`` stores a prepared answer for one question, and
``questions --clear-answer`` removes it again.

The honest rules, carried over from debriefs and goals:

- A prepared answer is your own words, stored as you wrote them.
  This module never generates an answer, so it cannot invent one for
  you — if a question has no prepared answer, the list says so
  instead of filling one in. Re-recording an answer replaces the
  earlier one, with the date it was updated.
- A question can be prepared before anyone has asked it: an answer
  stored for a question no debrief has recorded yet shows up as
  "not asked yet", not hidden, so the preparation is visible.
- Answers live in a small ``questions.json`` beside the application
  log (``--answers`` points somewhere else), the same pattern as
  ``goals.json``: a setting, not an event. Recording or clearing an
  answer never touches the log, appears in no history, and resets no
  clock.
- The bank is about the questions, not the live pipeline, so it
  counts debriefs on closed applications too: a question a company
  asked before rejecting you is still worth preparing for next time.
  What it does not do is guess: a question is grouped by its words
  (case and extra spaces ignored), and a debrief whose date will not
  parse is skipped, never fatal.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path


class QuestionsError(ValueError):
    """A prepared answer or answers file that cannot be used as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _normalize(question):
    """The grouping key for a question: words only, case ignored."""
    return " ".join(str(question).split()).lower()


def default_questions_path(log_path):
    """The answers file that belongs with a log: ``questions.json`` beside it."""
    return str(Path(log_path).parent / "questions.json")


def load_answers(path):
    """Read the answers file. A missing file means nothing prepared yet."""
    path = Path(path)
    if not path.exists():
        return {"answers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuestionsError(
            f"cannot read answers file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise QuestionsError(f"answers file {path} must hold one JSON object")
    answers = data.get("answers", {})
    if not isinstance(answers, dict):
        raise QuestionsError(
            f"answers file {path}: 'answers' must hold one JSON object")
    cleaned = {}
    for key, record in answers.items():
        if not isinstance(record, dict):
            raise QuestionsError(
                f"answers file {path}: answer for {key!r} must hold one "
                f"JSON object")
        question = record.get("question")
        answer = record.get("answer")
        updated_at = record.get("updated_at")
        if not isinstance(question, str) or not question.strip():
            raise QuestionsError(
                f"answers file {path}: answer for {key!r} has no question")
        if not isinstance(answer, str) or not answer.strip():
            raise QuestionsError(
                f"answers file {path}: answer for {key!r} has no answer text")
        if not isinstance(updated_at, str) or not updated_at.strip():
            raise QuestionsError(
                f"answers file {path}: answer for {key!r} has no updated date")
        cleaned[_normalize(key)] = {
            "question": question.strip(),
            "answer": answer.strip(),
            "updated_at": updated_at.strip(),
        }
    return {"answers": cleaned}


def save_answers(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return data


def validate_answer(question, answer):
    """Validate a prepared answer and return the cleaned form.

    Every problem is reported at once, like profile, offer, and
    debrief validation. The answer is the user's own words — this
    function checks that one was given, never what it says.
    """
    errors = []
    if question is None or (isinstance(question, str)
                            and not question.strip()):
        errors.append("question is required (the question to prepare for)")
    elif not isinstance(question, str):
        errors.append(f"question must be text, got {question!r}")
    if answer is None or (isinstance(answer, str)
                          and not answer.strip()):
        errors.append("answer is required (your prepared answer, in your "
                      "own words)")
    elif not isinstance(answer, str):
        errors.append(f"answer must be text, got {answer!r}")
    if errors:
        raise QuestionsError("; ".join(errors))
    return {"question": question.strip(), "answer": answer.strip()}


def record_answer(path, question, answer, today=None):
    """Store a prepared answer, replacing any earlier one for the same
    question. Returns (record, replaced)."""
    if today is None:
        today = datetime.date.today().isoformat()
    cleaned = validate_answer(question, answer)
    data = load_answers(path)
    key = _normalize(cleaned["question"])
    replaced = key in data["answers"]
    record = {"question": cleaned["question"],
              "answer": cleaned["answer"], "updated_at": today}
    data["answers"][key] = record
    save_answers(path, data)
    return record, replaced


def clear_answer(path, question):
    """Remove a prepared answer. Returns True when one was stored."""
    data = load_answers(path)
    key = _normalize(question)
    if key not in data["answers"]:
        return False
    del data["answers"][key]
    save_answers(path, data)
    return True


def collect_questions(entries, answers=None, only_unanswered=False,
                      contains=None):
    """Every question the debriefs recorded, grouped and most-asked first.

    Returns {"questions": [...], "summary": {...}}. Each row carries
    the question as first written, how many times it was asked, the
    companies that asked it, the date it was last asked, every
    occurrence (application, round, date), and the prepared answer
    when one is stored. Rows for questions with a stored answer but
    no debrief yet are included with ``asked`` 0 and ``asked_yet``
    false, so preparation done in advance stays visible. ``answers``
    is the dict ``load_answers`` returns (or None). ``today`` is not
    needed: nothing here is dated against today. A debrief whose date
    will not parse is skipped, never fatal, and a question listed
    twice in one debrief counts once.
    """
    answers = (answers or {}).get("answers", {})
    grouped = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for debrief in entry.get("debriefs") or []:
            if not isinstance(debrief, dict):
                continue
            when = _parse_date(debrief.get("date"))
            if when is None:
                continue
            questions = debrief.get("questions") or []
            if not isinstance(questions, list):
                continue
            seen_here = set()
            for question in questions:
                if not isinstance(question, str) or not question.strip():
                    continue
                key = _normalize(question)
                if key in seen_here:
                    continue
                seen_here.add(key)
                row = grouped.setdefault(key, {
                    "question": question.strip(),
                    "asked": 0,
                    "companies": set(),
                    "last_asked": None,
                    "occurrences": [],
                })
                row["asked"] += 1
                if entry.get("company"):
                    row["companies"].add(entry["company"])
                if row["last_asked"] is None or when.isoformat() > row["last_asked"]:
                    row["last_asked"] = when.isoformat()
                row["occurrences"].append({
                    "id": entry.get("id"),
                    "company": entry.get("company"),
                    "title": entry.get("title"),
                    "date": debrief["date"],
                    "round": debrief.get("round"),
                })
    for key, stored in answers.items():
        row = grouped.setdefault(key, {
            "question": stored["question"],
            "asked": 0,
            "companies": set(),
            "last_asked": None,
            "occurrences": [],
        })
        # The stored wording is the display form when the question was
        # never asked; when it was asked, the debrief wording stands.
        if not row["occurrences"]:
            row["question"] = stored["question"]
    rows = []
    for key, row in grouped.items():
        stored = answers.get(key)
        occurrence_sort = sorted(row["occurrences"],
                                 key=lambda occ: (occ["date"],
                                                  (occ["company"] or ""),
                                                  (occ["round"] or "")),
                                 reverse=True)
        rows.append({
            "question": row["question"],
            "asked": row["asked"],
            "asked_yet": row["asked"] > 0,
            "companies": sorted(row["companies"]),
            "last_asked": row["last_asked"],
            "occurrences": occurrence_sort,
            "answer": stored["answer"] if stored else None,
            "answer_updated_at": (stored["updated_at"] if stored else None),
            "answered": stored is not None,
        })
    if only_unanswered:
        rows = [row for row in rows if not row["answered"]]
    if contains:
        needle = _normalize(contains)
        rows = [row for row in rows
                if needle in _normalize(row["question"])]
    rows.sort(key=lambda row: (-row["asked"], row["answered"],
                               row["question"].lower()))
    total_asked = sum(row["asked"] for row in rows)
    answered = sum(1 for row in rows if row["answered"])
    return {
        "questions": rows,
        "summary": {
            "distinct": len(rows),
            "asked_total": total_asked,
            "answered": answered,
            "unanswered": len(rows) - answered,
            "repeated": sum(1 for row in rows if row["asked"] > 1),
        },
    }


def questions_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    if row["asked"] == 0:
        head = f"0x — {row['question']}  [not asked yet]"
    else:
        times = "time" if row["asked"] == 1 else "times"
        head = (f"{row['asked']}x — {row['question']}  "
                f"(asked {row['asked']} {times}")
        if row["companies"]:
            head += f", {', '.join(row['companies'])}"
        if row["last_asked"]:
            head += f", last asked {row['last_asked']}"
        head += ")"
    if row["answered"]:
        head += (f"\n      prepared answer "
                 f"(updated {row['answer_updated_at']}): {row['answer']}")
    else:
        head += "\n      [no prepared answer yet]"
    return head


def render_questions_markdown(data):
    """Human-readable question bank: most-asked first, answers shown."""
    rows = data["questions"]
    summary = data["summary"]
    question_word = ("question" if summary["distinct"] == 1
                     else "questions")
    time_word = "time" if summary["asked_total"] == 1 else "times"
    header = (f"Question bank — {summary['distinct']} distinct "
              f"{question_word}, asked {summary['asked_total']} {time_word}, "
              f"{summary['answered']} with a prepared answer, "
              f"{summary['unanswered']} still to prepare")
    lines = [header]
    if not rows:
        lines.append("No questions recorded yet — write up a debrief "
                     "with `--question ...` to start the bank, or prepare "
                     "an answer in advance with `questions --record "
                     "--question TEXT --answer TEXT`.")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    return "\n".join(lines) + "\n"
