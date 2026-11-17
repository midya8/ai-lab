"""Interview review: what the interview actually tested, against the posting.

``autojob review`` reads the debriefs already on file and checks them
against the posting each application went to: which skills the posting
asked for came up in the questions they actually asked, which never came
up, and whether the prep packet called the questions right.

Everything here is computed from records the user already made — the
debrief questions, the posting, the profile. The command records nothing
and never touches the log.

- A skill counts as probed when a debrief question names it, using the
  same word-boundary matching the posting parser uses: "SQL" in a
  question probes SQL, "MySQL" does not, and "stakeholder" on its own
  does not probe "Stakeholder Management". A question that names no
  posting skill is listed as off-script, verbatim — it may still have
  mattered, it just was not about a skill the posting asked for.
- Prep coverage is checked the same way: a skill-anchored likely
  question ("Tell me about a time you used SQL in your work.") counts as
  covered when any asked question names that skill. The walkthrough,
  why-this-role, and salary questions name no skill, so they cannot be
  checked from the debrief text and are listed as uncheckable, not
  scored as missed.
- A gap that came up — a skill the posting asked for that the profile
  does not have, named in a question — is flagged, because that is the
  question to prepare an honest answer for next time.
"""

from __future__ import annotations

import json
from pathlib import Path

from .match import match_profile
from .posting import find_skills, parse_posting
from .prep import likely_questions_with_skills
from .rank import POSTING_SUFFIXES
from .tracker import _norm


class ReviewError(ValueError):
    """A review that cannot be built as stated."""


def _parse_date(value):
    """A YYYY-MM-DD string as a date, or None when it is not one."""
    import datetime

    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _applied_date(entry):
    """The entry's applied date, or None when it is not parseable."""
    return _parse_date(entry.get("applied_at"))


def _review_order(entries):
    """Newest applied first; entries with no parseable date sort last.

    Same convention as ``search`` and ``export``: a stable sort keeps the
    log order (then id order) within one date.
    """
    import datetime

    ordered = sorted(entries, key=lambda e: e.get("id") or "")
    ordered.sort(key=lambda e: (_applied_date(e) is not None,
                                _applied_date(e) or datetime.date.min),
                 reverse=True)
    return ordered


def _load_postings(postings_dir, profile_skills):
    """(company, title) -> parsed posting, plus files that could not be read.

    Same folder convention as ``rank``: ``.md`` / ``.txt`` files, sorted
    by name; an unreadable file is reported in ``skipped``, never fatal.
    When two files parse to the same company and title, the first one
    wins — the folder is the user's, and the review says which file it
    used.
    """
    postings_dir = Path(postings_dir)
    files = sorted(
        (p for p in postings_dir.iterdir()
         if p.is_file() and p.suffix.lower() in POSTING_SUFFIXES),
        key=lambda p: p.name,
    ) if postings_dir.is_dir() else None
    if files is None:
        raise ValueError(f"not a directory: {postings_dir}")
    postings = {}
    skipped = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            continue
        posting = parse_posting(text, profile_skills)
        key = (_norm(posting.get("company")), _norm(posting.get("title")))
        if key not in postings:
            postings[key] = {"posting": posting, "file": path.name}
    return postings, skipped


def _asked_questions(entry):
    """(question, date, round) for every question in the entry's debriefs.

    A debrief that is not a dict, or a question that is not non-empty
    text, is skipped, never fatal — a hand-edited log still reviews.
    """
    asked = []
    debriefs = entry.get("debriefs")
    if not isinstance(debriefs, list):
        return asked
    for debrief in debriefs:
        if not isinstance(debrief, dict):
            continue
        questions = debrief.get("questions")
        if not isinstance(questions, list):
            continue
        for question in questions:
            if isinstance(question, str) and question.strip():
                asked.append({"question": question,
                              "date": debrief.get("date") or "",
                              "round": debrief.get("round") or ""})
    return asked


def _review_one(entry, profile, posting):
    """Review one application against its posting. Returns the row."""
    match = match_profile(profile, posting)
    asked = _asked_questions(entry)
    skill_names = (list(match["required_matched"])
                   + list(match["required_missing"])
                   + list(match["preferred_matched"])
                   + list(match["preferred_missing"]))
    probed = {}  # skill -> list of asked questions naming it
    off_script = []
    for item in asked:
        named = find_skills(item["question"], skill_names)
        if not named:
            off_script.append(item)
        for skill in named:
            probed.setdefault(skill, []).append(item)

    def skill_row(skill, kind):
        have = skill in (match["required_matched"]
                         + match["preferred_matched"])
        return {"skill": skill, "kind": kind, "have": have,
                "probed": skill in probed,
                "asked_in": probed.get(skill, [])}

    rows = ([skill_row(s, "required") for s in match["required_matched"]
             + match["required_missing"]]
            + [skill_row(s, "preferred") for s in match["preferred_matched"]
               + match["preferred_missing"]])
    required_probed = sum(1 for r in rows
                          if r["kind"] == "required" and r["probed"])
    preferred_probed = sum(1 for r in rows
                           if r["kind"] == "preferred" and r["probed"])
    gaps_probed = [r["skill"] for r in rows if not r["have"] and r["probed"]]

    likely = likely_questions_with_skills(profile, posting, match)
    anchored = [(q, s) for q, s in likely if s is not None]
    covered = [q for q, s in anchored if s in probed]
    not_covered = [q for q, s in anchored if s not in probed]
    uncheckable = [q for q, s in likely if s is None]
    prep = {"predicted": len(anchored),
            "covered": len(covered),
            "coverage": (round(len(covered) / len(anchored), 3)
                         if anchored else None),
            "covered_questions": covered,
            "not_covered": not_covered,
            "uncheckable": uncheckable}

    return {
        "id": entry.get("id"),
        "company": entry.get("company"),
        "title": entry.get("title"),
        "status": entry.get("status"),
        "applied_at": entry.get("applied_at"),
        "n_debriefs": (len(entry.get("debriefs"))
                       if isinstance(entry.get("debriefs"), list) else 0),
        "n_questions": len(asked),
        "skills_probed": required_probed + preferred_probed,
        "skills_total": len(rows),
        "required_probed": required_probed,
        "required_total": (len(match["required_matched"])
                           + len(match["required_missing"])),
        "preferred_probed": preferred_probed,
        "preferred_total": (len(match["preferred_matched"])
                            + len(match["preferred_missing"])),
        "gaps_probed": gaps_probed,
        "off_script": off_script,
        "prep": prep,
        "skill_rows": rows,
    }


def collect_review(entries, profile, postings_dir, only_id=None):
    """Review every application with debriefs against its posting.

    Returns ``{"reviewed": [...], "skipped": [...], "summary": {...}}``.
    An application is skipped, with the reason, when it has no debriefs
    or its posting is not in the folder; a debrief with no questions
    still reviews, with a note that nothing was asked on record. An
    ``only_id`` that matches no application is an error, not an empty
    review. Non-dict log lines are skipped, never fatal.
    """
    entries = [e for e in (entries or []) if isinstance(e, dict)]
    if only_id is not None:
        wanted = [e for e in entries if e.get("id") == only_id]
        if not wanted:
            raise ReviewError(f"no application with id {only_id!r}")
        entries = wanted
    profile_skills = [s["name"] for s in profile.get("skills", [])]
    postings, skipped_files = _load_postings(postings_dir, profile_skills)
    reviewed, skipped = [], []
    for entry in _review_order(entries):
        debriefs = entry.get("debriefs")
        if not isinstance(debriefs, list) or not debriefs:
            skipped.append({"id": entry.get("id"),
                            "company": entry.get("company"),
                            "title": entry.get("title"),
                            "reason": "no debriefs recorded"})
            continue
        key = (_norm(entry.get("company")), _norm(entry.get("title")))
        found = postings.get(key)
        if found is None:
            skipped.append({"id": entry.get("id"),
                            "company": entry.get("company"),
                            "title": entry.get("title"),
                            "reason": "no posting in the folder"})
            continue
        row = _review_one(entry, profile, found["posting"])
        row["posting_file"] = found["file"]
        reviewed.append(row)
    for item in skipped_files:
        skipped.append({"id": None, "company": None, "title": None,
                        "reason": f"unreadable posting file {item['file']}: "
                                  f"{item['reason']}"})
    gaps = {}
    for row in reviewed:
        for skill in row["gaps_probed"]:
            gaps.setdefault(skill, []).append(row["company"])
    summary = {
        "reviewed": len(reviewed),
        "skipped": len(skipped),
        "skills_probed": sum(r["skills_probed"] for r in reviewed),
        "skills_total": sum(r["skills_total"] for r in reviewed),
        "n_questions": sum(r["n_questions"] for r in reviewed),
        "off_script_total": sum(len(r["off_script"]) for r in reviewed),
        "gaps_probed": {skill: sorted(set(companies))
                        for skill, companies in sorted(gaps.items())},
        "prep_predicted": sum(r["prep"]["predicted"] for r in reviewed),
        "prep_covered": sum(r["prep"]["covered"] for r in reviewed),
        "prep_coverage": (
            round(sum(r["prep"]["covered"] for r in reviewed)
                  / sum(r["prep"]["predicted"] for r in reviewed), 3)
            if sum(r["prep"]["predicted"] for r in reviewed) else None),
    }
    return {"reviewed": reviewed, "skipped": skipped, "summary": summary}


def review_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_skill_row(row):
    probed = "asked" if row["probed"] else "not asked"
    have = "" if row["have"] else " (gap — not in your profile)"
    asked = ""
    if row["asked_in"]:
        asked = " — " + "; ".join(
            f"\"{a['question']}\""
            + (f" ({a['round']}, {a['date']})" if a["round"] or a["date"]
               else "")
            for a in row["asked_in"])
    return f"- {row['skill']} [{row['kind']}] — {probed}{have}{asked}"


def render_review_markdown(data):
    """Render the review as Markdown. Facts only — see module docstring."""
    summary = data["summary"]
    lines = [f"# Interview review — {summary['reviewed']} application"
             f"{'s' if summary['reviewed'] != 1 else ''} reviewed", ""]
    lines.append(f"- Questions on record: {summary['n_questions']} "
                 f"({summary['off_script_total']} off-script)")
    lines.append(f"- Posting skills probed: {summary['skills_probed']} "
                 f"of {summary['skills_total']}")
    if summary["prep_predicted"]:
        coverage = summary["prep_coverage"]
        lines.append(f"- Prep called {summary['prep_covered']} of "
                     f"{summary['prep_predicted']} skill questions "
                     f"({coverage:.0%})" if coverage is not None else
                     f"- Prep called {summary['prep_covered']} of "
                     f"{summary['prep_predicted']} skill questions")
    if summary["gaps_probed"]:
        lines.append("- Gaps that came up: " + ", ".join(
            f"{skill} ({', '.join(companies)})"
            for skill, companies in summary["gaps_probed"].items()))
    lines.append("")
    if not data["reviewed"]:
        lines.append("Nothing reviewed: no application in the log has a "
                     "debrief and a posting in the folder.")
        lines.append("")
    for row in data["reviewed"]:
        lines.append(f"## {row['company']} — {row['title']} ({row['id']})")
        lines.append("")
        lines.append(f"- {row['n_questions']} question"
                     f"{'s' if row['n_questions'] != 1 else ''} across "
                     f"{row['n_debriefs']} debrief"
                     f"{'s' if row['n_debriefs'] != 1 else ''}"
                     f" (posting: {row['posting_file']})")
        lines.append(f"- Skills probed: {row['skills_probed']} of "
                     f"{row['skills_total']} — required "
                     f"{row['required_probed']}/{row['required_total']}, "
                     f"preferred "
                     f"{row['preferred_probed']}/{row['preferred_total']}")
        if row["gaps_probed"]:
            lines.append("- Gaps that came up: "
                         + ", ".join(row["gaps_probed"])
                         + " — prepare an honest answer for next time.")
        prep = row["prep"]
        if prep["predicted"]:
            lines.append(f"- Prep: {prep['covered']} of "
                         f"{prep['predicted']} skill questions came up"
                         + (f" ({prep['coverage']:.0%})"
                            if prep["coverage"] is not None else "")
                         + (f"; not asked: "
                            + "; ".join(prep["not_covered"])
                            if prep["not_covered"] else ""))
        if prep["uncheckable"]:
            lines.append("- Prep questions that cannot be checked from the "
                         "debrief text (they name no skill): "
                         + "; ".join(prep["uncheckable"]))
        lines.append("")
        for skill_row in row["skill_rows"]:
            lines.append(_fmt_skill_row(skill_row))
        if row["off_script"]:
            lines.append("")
            lines.append("Off-script questions (no posting skill named):")
            for item in row["off_script"]:
                lines.append(f"- \"{item['question']}\""
                             + (f" ({item['round']}, {item['date']})"
                                if item["round"] or item["date"] else ""))
        lines.append("")
    if data["skipped"]:
        lines.append("## Skipped")
        lines.append("")
        for item in data["skipped"]:
            who = item.get("company") or item.get("id") or "a posting file"
            title = f" — {item['title']}" if item.get("title") else ""
            lines.append(f"- {who}{title}: {item['reason']}")
        lines.append("")
    return "\n".join(lines)
