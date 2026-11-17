"""Full-text search across the application log.

``autojob search --query TEXT`` finds the applications that mention
something — a person, a company, a skill, a question an interviewer
asked, a note you wrote yourself. The log holds a lot by now
(contacts, interviews, debriefs, offers, follow-ups, rejection
reasons, onboarding tasks), and scrolling through ``list`` to find one detail gets old.

The honest rules, carried over from the other modules:

- The query is split into words, and every word must appear
  somewhere in the application — words may match in different fields,
  so "Priya video" finds the application whose hiring-manager round
  was with Priya Nair on video. Matching is case-insensitive and
  extra spaces are ignored. There is no ranking and no fuzzy match:
  a word either appears in the recorded text or it does not, and the
  result says which fields it appeared in, so you can see why a row
  matched instead of trusting a score.
- Search is read-only: it records nothing, changes nothing, and
  never touches the log. It only reads what the other commands
  already recorded, word for word — including the resume file
  recorded for an application, so a version name is searchable
  like any other recorded fact.
- Nothing is guessed. A skill list the log never recorded is simply
  not searched; an entry with no parseable ``applied_at`` sorts last
  instead of being dated by assumption; a non-dict entry or a record
  with a wrong shape is skipped, never fatal.
- ``--status`` and ``--source`` narrow the rows first, the same as
  ``export``: a filter value that is not a real status or channel is
  refused, not silently matched to nothing. An empty query is
  refused too — searching for nothing would list everything, which
  is what ``list`` is for.
"""

from __future__ import annotations

import datetime
import json

from .sources import SOURCE_CHOICES
from .tracker import STATUSES

# The order matched fields are listed in, so a row reads the same way
# every time regardless of which fields matched.
FIELD_ORDER = ("id", "company", "title", "status", "source", "referrer",
               "skills", "deadline", "rejection", "resume", "contact",
               "interview", "offer", "debrief", "followup", "note",
               "onboarding")

SNIPPET_LEN = 160


class SearchError(ValueError):
    """A query or filter that cannot be used as stated."""


def _text(*parts):
    """Join the non-empty text parts of one record, or "" when none."""
    cleaned = [str(part).strip() for part in parts
               if isinstance(part, str) and str(part).strip()]
    return " · ".join(cleaned)


def _skill_text(entry):
    """The skills a posting asked for and which ones the profile had,
    as one line of text, or "" when the log never recorded them."""
    groups = []
    for key, label in (("required_matched", "required matched"),
                       ("required_missing", "required missing"),
                       ("preferred_matched", "preferred matched"),
                       ("preferred_missing", "preferred missing")):
        skills = entry.get(key) or []
        if isinstance(skills, list) and skills:
            named = [str(skill) for skill in skills
                     if isinstance(skill, str) and skill.strip()]
            if named:
                groups.append(f"{label}: {', '.join(named)}")
    return "; ".join(groups)


def searchable_fields(entry):
    """Every searchable (field, text) pair for one log entry.

    One pair per record for the list fields, so a match can name the
    contact, interview, offer, debrief, or follow-up it came from
    instead of just the kind of thing. Empty fields are left out.
    """
    if not isinstance(entry, dict):
        return []
    fields = []
    for key in ("id", "company", "title", "status", "source", "referrer",
                "deadline"):
        text = _text(entry.get(key))
        if text:
            fields.append((key, text))
    skills = _skill_text(entry)
    if skills:
        fields.append(("skills", skills))
    rejection = entry.get("rejection") or {}
    if isinstance(rejection, dict):
        # Stored as "category" (see rejections.validate_rejection);
        # "reason" is accepted too, for a hand-edited log.
        reason = rejection.get("category") or rejection.get("reason")
        text = _text(reason, rejection.get("note"))
        if text:
            fields.append(("rejection", text))
    resume = entry.get("resume") or {}
    if isinstance(resume, dict):
        text = _text(resume.get("path"), resume.get("note"))
        if text:
            fields.append(("resume", text))
    for contact in entry.get("contacts") or []:
        if not isinstance(contact, dict):
            continue
        text = _text(contact.get("name"), contact.get("role"),
                     contact.get("channel"), contact.get("note"))
        if text:
            fields.append(("contact", text))
    for interview in entry.get("interviews") or []:
        if not isinstance(interview, dict):
            continue
        text = _text(interview.get("round"), interview.get("interviewer"),
                     interview.get("mode"), interview.get("note"))
        if text:
            fields.append(("interview", text))
    for offer in entry.get("offers") or []:
        if not isinstance(offer, dict):
            continue
        text = _text(offer.get("mode"), offer.get("currency"),
                     offer.get("note"))
        if text:
            fields.append(("offer", text))
    for debrief in entry.get("debriefs") or []:
        if not isinstance(debrief, dict):
            continue
        questions = debrief.get("questions") or []
        question_text = " · ".join(
            str(q) for q in questions
            if isinstance(q, str) and q.strip())
        text = _text(debrief.get("round"), debrief.get("went_well"),
                     debrief.get("improve"), question_text or None)
        if text:
            fields.append(("debrief", text))
    for event in entry.get("history") or []:
        if not isinstance(event, dict):
            continue
        if event.get("kind") != "followup":
            continue
        text = _text(event.get("note"))
        if text:
            fields.append(("followup", text))
    for note in entry.get("notes") or []:
        if not isinstance(note, dict):
            continue
        text = _text(note.get("tag"), note.get("text"))
        if text:
            fields.append(("note", text))
    for task in entry.get("onboarding") or []:
        if not isinstance(task, dict):
            continue
        text = _text(task.get("task"), task.get("category"))
        if text:
            fields.append(("onboarding", text))
    return fields


def validate_query(query):
    """Validate a search query and return (query, terms).

    The query is required and must hold at least one word; the terms
    are its words, casefolded, in order. An empty or non-text query
    is refused.
    """
    if query is None or (isinstance(query, str) and not query.strip()):
        raise SearchError("query is required (the words to search for)")
    if not isinstance(query, str):
        raise SearchError(f"query must be text, got {query!r}")
    cleaned = " ".join(query.split())
    return cleaned, cleaned.casefold().split()


def search_entries(entries, query, status=None, source=None):
    """Search the log. Returns {"query", "terms", "results", "summary"}.

    Every term in the query must appear somewhere in an application
    (case-insensitive substring; terms may match in different
    fields). Each result carries the application, the matched fields
    in canonical order, and one snippet per matching record — the
    record's text, truncated to 160 characters, so the match is
    visible without opening the log. Results are newest applied
    first; entries with no parseable ``applied_at`` sort last.
    ``status`` and ``source`` narrow the rows first; a bad filter
    value raises SearchError. A non-dict entry is skipped, never
    fatal.
    """
    cleaned, terms = validate_query(query)
    if status is not None and status not in STATUSES:
        raise SearchError(
            f"status must be one of {', '.join(STATUSES)}, got {status!r}")
    if source is not None and source not in SOURCE_CHOICES:
        raise SearchError(
            f"source must be one of {', '.join(SOURCE_CHOICES)}, "
            f"got {source!r}")
    scored = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if status is not None and entry.get("status") != status:
            continue
        if source is not None and entry.get("source") != source:
            continue
        fields = searchable_fields(entry)
        matches = []
        matched_labels = set()
        for field, text in fields:
            lowered = text.casefold()
            if any(term in lowered for term in terms):
                matched_labels.add(field)
                snippet = (text if len(text) <= SNIPPET_LEN
                           else text[:SNIPPET_LEN - 3] + "...")
                matches.append({"field": field, "text": snippet})
        if not matched_labels:
            continue
        # Every term must appear somewhere in the application; the
        # per-field loop above only proves each matched field holds
        # at least one term.
        joined = " ".join(text for _, text in fields).casefold()
        if not all(term in joined for term in terms):
            continue
        applied = None
        raw = entry.get("applied_at")
        if isinstance(raw, str):
            try:
                applied = datetime.date.fromisoformat(raw)
            except ValueError:
                applied = None
        order = FIELD_ORDER.index
        labels = sorted(matched_labels, key=lambda label: order(label))
        scored.append(({
            "id": entry.get("id"),
            "company": entry.get("company"),
            "title": entry.get("title"),
            "status": entry.get("status"),
            "applied_at": entry.get("applied_at"),
            "match_score": entry.get("match_score"),
            "fields": labels,
            "matches": matches,
        }, applied))
    # Newest applied first; entries with no parseable date sort last.
    # Stable sort keeps the log order (then id order) within one date.
    scored.sort(key=lambda pair: pair[0].get("id") or "")
    scored.sort(key=lambda pair: (pair[1] is not None,
                                  pair[1] or datetime.date.min),
               reverse=True)
    return {
        "query": cleaned,
        "terms": terms,
        "results": [row for row, _ in scored],
        "summary": {
            "matched": len(scored),
            "total": sum(1 for entry in entries
                         if isinstance(entry, dict)),
            "status": status,
            "source": source,
        },
    }


def search_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def render_search_markdown(data):
    """Human-readable search results: one block per application,
    with the matched fields shown so the match is visible."""
    summary = data["summary"]
    query = data["query"]
    header = (f"Search — {summary['matched']} of {summary['total']} "
              f"application(s) matching {query!r}")
    if summary["status"]:
        header += f", status {summary['status']}"
    if summary["source"]:
        header += f", source {summary['source']}"
    lines = [header]
    if not data["results"]:
        lines.append(f"No applications mention {query!r} — check the "
                     f"spelling, or try fewer words (every word in the "
                     f"query has to appear somewhere in the application).")
        return "\n".join(lines) + "\n"
    for row in data["results"]:
        head = f"{row['id']} — {row['company']}, {row['title']}"
        details = []
        if row["status"]:
            details.append(row["status"])
        if row["applied_at"]:
            details.append(f"applied {row['applied_at']}")
        if row["match_score"] is not None:
            details.append(f"score {row['match_score']}")
        if details:
            head += f" ({'; '.join(details)})"
        lines.append(head)
        for match in row["matches"]:
            lines.append(f"  {match['field']}: {match['text']}")
    return "\n".join(lines) + "\n"
