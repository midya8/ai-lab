"""Rank a folder of postings by fit: where to apply next.

``rank`` reads every ``.md`` / ``.txt`` posting in a directory, scores each
one against the profile with the same transparent match as ``match``, and
lists them best-first so the strongest application comes first. Postings
already in the application log are left out by default — ranking is about
where to apply *next* — unless ``--include-applied`` is given.

A posting that cannot be read (missing file, not UTF-8) is reported in
``skipped`` and never crashes the ranking; a file with no recognisable
skills scores ``None`` and sorts last, labelled as unmeasurable rather
than given a made-up zero. Sorting is by score, then required coverage,
then title, so ties break the same way every run.
"""

from __future__ import annotations

import json
from pathlib import Path

from .match import match_profile
from .posting import parse_posting
from .tracker import _norm

POSTING_SUFFIXES = (".md", ".txt")
TOP_GAPS = 3


def _applied_map(entries):
    """(normalised company, normalised title) -> entry id, for the log."""
    applied = {}
    for entry in entries:
        key = (_norm(entry.get("company")), _norm(entry.get("title")))
        applied.setdefault(key, entry.get("id"))
    return applied


def rank_postings(profile, postings_dir, entries=(), min_score=None,
                  include_applied=False):
    """Rank every posting file in ``postings_dir`` against ``profile``.

    Returns ``{"ranked": [...], "skipped": [...], ...}``. Each ranked row
    carries the file name, title, company, score, verdict, required /
    preferred coverage and match lists, the top required gaps, and
    whether the posting is already in the log (with its application id).
    Rows already applied are dropped unless ``include_applied``; rows
    below ``min_score`` are dropped (an unmeasurable ``None`` score never
    passes a threshold). Unreadable files land in ``skipped`` with a
    reason; they are not ranked and not silently ignored.
    """
    postings_dir = Path(postings_dir)
    files = sorted(
        (p for p in postings_dir.iterdir()
         if p.is_file() and p.suffix.lower() in POSTING_SUFFIXES),
        key=lambda p: p.name,
    ) if postings_dir.is_dir() else None
    if files is None:
        raise ValueError(f"not a directory: {postings_dir}")
    applied = _applied_map(entries)
    profile_skills = [s["name"] for s in profile.get("skills", [])]
    ranked, skipped = [], []
    n_scored = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            continue
        posting = parse_posting(text, profile_skills)
        match = match_profile(profile, posting)
        n_scored += 1
        key = (_norm(posting.get("company")), _norm(posting.get("title")))
        applied_id = applied.get(key)
        row = {
            "file": path.name,
            "title": posting.get("title") or "unknown role",
            "company": posting.get("company") or "unknown company",
            "score": match["score"],
            "verdict": match["verdict"],
            "required_coverage": match["required_coverage"],
            "preferred_coverage": match["preferred_coverage"],
            "required_matched": match["required_matched"],
            "required_missing": match["required_missing"],
            "preferred_matched": match["preferred_matched"],
            "preferred_missing": match["preferred_missing"],
            "top_gaps": match["required_missing"][:TOP_GAPS],
            "applied": applied_id is not None,
            "applied_id": applied_id,
        }
        if row["applied"] and not include_applied:
            continue
        if min_score is not None:
            if row["score"] is None or row["score"] < min_score:
                continue
        ranked.append(row)

    def sort_key(row):
        score = row["score"]
        coverage = row["required_coverage"]
        return (
            0 if score is not None else 1,
            -(score if score is not None else 0.0),
            -(coverage if coverage is not None else -1.0),
            row["title"].lower(),
            row["file"],
        )

    ranked.sort(key=sort_key)
    return {
        "ranked": ranked,
        "skipped": skipped,
        "n_scored": n_scored,
        "n_files": len(files),
        "n_skipped": len(skipped),
        "min_score": min_score,
        "include_applied": include_applied,
    }


def rank_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _score_text(row):
    if row["score"] is None:
        return "unmeasurable (no skills found)"
    return f"{row['score']:.3f} ({row['verdict']})"


def render_rank_markdown(data):
    """Human-readable ranking, best first, with the next application named."""
    ranked = data["ranked"]
    lines = [f"Posting ranking — {len(ranked)} ranked "
             f"({data['n_files']} files, {data['n_skipped']} skipped)"]
    if data["min_score"] is not None:
        lines[0] += f", min score {data['min_score']}"
    lines.append("")
    if not ranked:
        lines.append("No postings to rank"
                     + (" above the threshold" if data["min_score"] is not None else "")
                     + ".")
        if data["skipped"]:
            lines.append("")
            lines.append("Skipped files (could not be read):")
            for item in data["skipped"]:
                lines.append(f"  {item['file']}: {item['reason']}")
        return "\n".join(lines) + "\n"
    next_up = next((r for r in ranked if not r["applied"]), None)
    if next_up is not None:
        lines.append(f"Best next application: {next_up['title']} — "
                     f"{next_up['company']} "
                     f"(score {_score_text(next_up)})")
    else:
        lines.append("Best next application: none — every ranked posting "
                     "is already in the log.")
    lines.append("")
    for i, row in enumerate(ranked, 1):
        req = (f"{len(row['required_matched'])}/"
               f"{len(row['required_matched']) + len(row['required_missing'])}")
        line = (f"  {i}. {row['title']} — {row['company']}  "
                f"score {_score_text(row)}  required {req}")
        if row["top_gaps"]:
            line += f"  top gaps: {', '.join(row['top_gaps'])}"
        if row["applied"]:
            line += f"  [already applied as {row['applied_id']}]"
        lines.append(line)
    if data["skipped"]:
        lines.append("")
        lines.append("Skipped files (could not be read):")
        for item in data["skipped"]:
            lines.append(f"  {item['file']}: {item['reason']}")
    return "\n".join(lines) + "\n"
