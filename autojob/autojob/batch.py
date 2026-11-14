"""Batch apply: one packet per posting in a folder, best fit first.

``batch`` is ``rank`` plus ``apply`` in one pass. It scores every
``.md`` / ``.txt`` posting in a directory against the profile, best
first, and builds a full application packet (cover letter, match
report, review notes, interview prep, tailored resume) for each one
that clears the bar, logging each to the tracker as it goes. It is
still drafts only: nothing is submitted anywhere, and every packet
is for a human to review before sending.

A posting is skipped — never guessed at — when it cannot be read,
when it has no recognisable skills (unmeasurable, not a zero), when
it scores below ``--min-score``, when it is already in the log, or
when ``--limit`` has been reached. Skips are reported with the
reason, so a batch that applies nothing still tells you why. A
second file for the same company and title in the same folder is
skipped as already applied once the first one lands, and a tracker
write failure stops the batch: the packets already written stay
logged, the rest are reported as not attempted, and the error is
in the result instead of being swallowed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .letter import draft_letter, review_notes
from .match import match_profile
from .posting import parse_posting
from .prep import build_prep, render_prep_markdown
from .rank import rank_postings
from .report import match_json
from .sources import validate_source, SourceError
from .tailor import build_tailor, render_tailor_markdown
from .tracker import TrackerError, add_application, read_log, slug


class BatchError(ValueError):
    """Bad batch arguments: min-score, limit, or source/referrer."""


def packet_dir_for(out_dir, posting):
    """Where ``apply``/``batch`` put one posting's packet."""
    return (Path(out_dir)
            / f"{slug(posting.get('company'))}-{slug(posting.get('title'))}")


def build_packet_files(profile, posting, match):
    """The five files every application packet holds."""
    return {
        "cover-letter.md": draft_letter(profile, posting, match),
        "match.json": match_json(posting, match),
        "review-notes.md": review_notes(profile, posting, match),
        "interview-prep.md": render_prep_markdown(
            build_prep(profile, posting, match)),
        "tailored-resume.md": render_tailor_markdown(
            build_tailor(profile, posting, match)),
    }


def batch_apply(profile, postings_dir, out_dir, log_path, min_score=0.0,
                limit=None, source=None, referrer=None, dry_run=False,
                today=None):
    """Apply to a folder of postings, best fit first.

    Returns ``{\"applied\": [...], \"skipped\": [...], \"counts\": {...},
    \"dry_run\": bool, \"error\": str | None}``. ``source``/``referrer``
    are validated before anything is written; a bad ``min_score`` or
    ``limit`` raises :class:`BatchError` before anything is read.
    ``log_path`` is read up front for the already-applied check, and
    the in-memory map is updated after each successful apply, so a
    duplicate posting later in the same folder is skipped, not
    double-logged. A tracker write failure stops the batch and is
    reported in ``error``; packets already written stay logged.
    """
    if not 0.0 <= min_score <= 1.0:
        raise BatchError(f"--min-score must be between 0 and 1, got {min_score}")
    if limit is not None and limit < 1:
        raise BatchError(f"--limit must be at least 1, got {limit}")
    if source is not None:
        try:
            cleaned = validate_source(source, referrer)
        except SourceError as exc:
            raise BatchError(str(exc)) from exc
        source = cleaned["source"]
        referrer = cleaned["referrer"]
    entries = read_log(log_path)
    ranked = rank_postings(profile, postings_dir, entries,
                           include_applied=True)
    profile_skills = [s["name"] for s in profile.get("skills", [])]
    applied, skipped = [], []
    applied_ids = {
        (entry.get("company", "").strip().lower(),
         entry.get("title", "").strip().lower()): entry.get("id")
        for entry in entries
    }
    dry_run_keys = set()
    error = None
    for row in ranked["ranked"]:
        key = (row["company"].strip().lower(), row["title"].strip().lower())
        reason = None
        if row["score"] is None:
            reason = "no recognisable skills (unmeasurable)"
        elif key in applied_ids:
            reason = f"already applied as {applied_ids[key]}"
        elif key in dry_run_keys:
            reason = "already applied in this batch (dry run)"
        elif row["score"] < min_score:
            reason = (f"score {row['score']:.3f} below "
                      f"--min-score {min_score:.3f}")
        elif limit is not None and len(applied) >= limit:
            reason = f"beyond --limit {limit}"
        if reason is not None:
            skipped.append({**row, "reason": reason})
            continue
        path = Path(postings_dir) / row["file"]
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({**row, "reason": str(exc)})
            continue
        posting = parse_posting(text, profile_skills)
        match = match_profile(profile, posting)
        packet = packet_dir_for(out_dir, posting)
        files = build_packet_files(profile, posting, match)
        if dry_run:
            applied.append({**row, "id": None, "packet_dir": str(packet),
                            "dry_run": True})
            dry_run_keys.add(key)
            continue
        packet.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (packet / name).write_text(content, encoding="utf-8")
        try:
            entry = add_application(
                log_path,
                company=posting.get("company") or "unknown company",
                title=posting.get("title") or "unknown role",
                source=source,
                match_score=match["score"],
                packet_dir=str(packet),
                today=today,
                referrer=referrer,
                required_matched=match["required_matched"],
                required_missing=match["required_missing"],
                preferred_matched=match["preferred_matched"],
                preferred_missing=match["preferred_missing"],
            )
        except TrackerError as exc:
            error = str(exc)
            break
        applied.append({**row, "id": entry["id"],
                        "packet_dir": str(packet)})
        applied_ids[key] = entry["id"]
    for row in ranked["skipped"]:
        skipped.append({**row, "company": "unknown company",
                        "title": "unknown role", "score": None,
                        "reason": row["reason"]})
    result = {
        "applied": applied,
        "skipped": skipped,
        "counts": {
            "applied": len(applied),
            "skipped": len(skipped),
            "scored": len(ranked["ranked"]),
            "files": len(ranked["ranked"]) + len(ranked["skipped"]),
        },
        "dry_run": dry_run,
        "error": error,
    }
    return result


def batch_json(result):
    """The batch result as JSON, keys sorted for a stable diff."""
    return json.dumps(result, indent=2, sort_keys=True)


def render_batch_markdown(result):
    """The batch result as one readable page."""
    counts = result["counts"]
    lines = [f"Batch apply — {counts['applied']} applied, "
             f"{counts['skipped']} skipped "
             f"({counts['scored']} scored, {counts['files']} files)"]
    if result["dry_run"]:
        lines[0] += " (dry run — nothing written)"
    if result["applied"]:
        lines.append("")
        lines.append("Applied:")
        for row in result["applied"]:
            score = (f"{row['score']:.3f}" if row["score"] is not None
                     else "unmeasurable")
            lines.append(f"- {row['company']} — {row['title']} "
                         f"(score {score}, {row['id'] or 'not logged'})")
    if result["skipped"]:
        lines.append("")
        lines.append("Skipped:")
        for row in result["skipped"]:
            if row.get("company") not in (None, "unknown company"):
                label = (f"{row['company']} — {row['title']} "
                         f"({row.get('file', '')})")
            else:
                label = row.get("file", row["company"])
            lines.append(f"- {label}: {row['reason']}")
    if result["error"]:
        lines.append("")
        lines.append(f"Stopped early: {result['error']}")
    if not result["applied"] and not result["skipped"]:
        lines.append("")
        lines.append("No postings found.")
    return "\n".join(lines) + "\n"
