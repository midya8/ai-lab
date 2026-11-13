"""Application sources: where each application came from, and what worked.

``sources --record`` stores how you found a posting — one of six fixed
channels — plus, for a referral, the name of the person who referred you.
``sources`` on its own answers the only question the field exists to
answer: which channels actually lead anywhere. Each source shows how many
applications came through it, how many of those reached an interview, and
how many were hired, with the rate stated as a share of that source's own
applications (``n=`` is always shown, because a 1.00 rate on one
application means nothing). Referrals are listed separately, person by
person, so you can see who to thank — and who to follow up with.

The honest rules, carried over from the other modules:

- A source is a fact you supply, about how you found the posting. It is
  stored as fields on the tracker entry (see ``tracker.record_source``),
  not as a history event: finding a posting is not something that
  happened to the application, so recording a source does not reset the
  follow-up or stale clocks, and it does not appear in the timeline feed.
  Re-recording replaces the source — you may learn later that the
  posting you thought came from a job board actually came from a
  recruiter — so this is a set, not an append. Recording a source is
  refused on a terminal application, the same rule as a closing date:
  the source belongs on the application while it is live.
- The source is one of six fixed values (``job-board``, ``company-site``,
  ``referral``, ``recruiter``, ``networking``, ``other``). Free text was
  accepted before v0.18 and is still stored verbatim on the entry, but
  the summary only counts values on the list: an old free-text value is
  treated as unrecorded, never fatal, and never guessed into a channel.
  A referral without a referrer is refused — the point of tracking a
  referral is knowing who referred you, and a name recorded later is
  better than a referral you cannot follow up on. A referrer may also
  be recorded for any other source (a recruiter's name, the person who
  mentioned the role), but it is never required there.
- "Reached interview" comes from the tracker's recorded history: any
  status move to ``interview``, ``offer``, or ``hired``, or a current
  status in that set. An application that reached the interview stage
  and was then rejected still counts — it got that far, and the source
  that got it there is the one that worked. The rate is a description
  of your own log, not a prediction: it says nothing about what the
  next application through that channel will do.
- Applications with no source recorded are counted, not hidden: the
  header says how many carry no source, so an unrecorded source is
  visible work, not a silent gap.
"""

from __future__ import annotations

import datetime
import json

from .analytics import reached_stages

SOURCE_CHOICES = (
    "job-board",
    "company-site",
    "referral",
    "recruiter",
    "networking",
    "other",
)

INTERVIEW_STAGES = ("interview", "offer", "hired")


class SourceError(ValueError):
    """A source that cannot be recorded as stated."""


def validate_source(source, referrer=None):
    """Validate a source and return the cleaned form.

    Every problem is reported at once, like profile, offer, interview,
    contact, and rejection validation. The source must be one of
    SOURCE_CHOICES (surrounding whitespace and case are forgiven, and
    the stored value is the canonical lowercase form); the referrer is
    free text, kept verbatim apart from surrounding whitespace, and
    required when the source is ``referral``.
    """
    errors = []
    cleaned_source = None
    if source is None or (isinstance(source, str) and not source.strip()):
        errors.append("source is required")
    elif not isinstance(source, str):
        errors.append(f"source must be one of {', '.join(SOURCE_CHOICES)}, "
                      f"got {source!r}")
    else:
        cleaned_source = source.strip().lower()
        if cleaned_source not in SOURCE_CHOICES:
            errors.append(f"source must be one of "
                          f"{', '.join(SOURCE_CHOICES)}, "
                          f"got {source!r}")
    if referrer is not None and not isinstance(referrer, str):
        errors.append(f"referrer must be text, got {referrer!r}")
    cleaned_referrer = (referrer.strip()
                        if isinstance(referrer, str) and referrer.strip()
                        else None)
    if (cleaned_source == "referral" and cleaned_referrer is None):
        errors.append("a referral needs a referrer: who referred you?")
    if errors:
        raise SourceError("; ".join(errors))
    return {"source": cleaned_source, "referrer": cleaned_referrer}


def _recorded_source(entry):
    """The entry's recorded source, or None.

    A stored source outside SOURCE_CHOICES (free text recorded before
    v0.18, or a hand-edited log) is treated as unrecorded, never fatal.
    """
    source = entry.get("source")
    if not isinstance(source, str):
        return None
    cleaned = source.strip().lower()
    if cleaned not in SOURCE_CHOICES:
        return None
    referrer = entry.get("referrer")
    if not isinstance(referrer, str) or not referrer.strip():
        referrer = None
    else:
        referrer = referrer.strip()
    return {"source": cleaned, "referrer": referrer}


def _reached_interview(entry):
    """True when the application reached interview, offer, or hired."""
    return bool(reached_stages(entry) & set(INTERVIEW_STAGES))


def _counted(mapping):
    """A {name: count} dict ordered by count desc, then name."""
    return dict(sorted(mapping.items(), key=lambda pair: (-pair[1], pair[0])))


def collect_sources(entries, source=None):
    """Per-source conversion plus the referral list.

    Returns {"sources": [...], "referrals": [...], "counts": {...}}.
    Each source row carries the source, how many applications came
    through it, how many reached an interview and the rate among that
    source's own applications, how many were hired, and the companies
    (sorted, unique). Referrals list every application whose source is
    ``referral``, newest applied first, with the referrer, company,
    title, status, applied date, and whether it reached an interview.
    ``source`` narrows the rows to one channel; an unknown channel
    raises ValueError. Entries that are not dicts are skipped, never
    fatal.
    """
    if source is not None and source not in SOURCE_CHOICES:
        raise ValueError(
            f"source must be one of {', '.join(SOURCE_CHOICES)}, "
            f"got {source!r}")
    by_source = {}
    referrals = []
    no_source = 0
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        total += 1
        recorded = _recorded_source(entry)
        if recorded is None:
            no_source += 1
            continue
        name = recorded["source"]
        row = by_source.get(name)
        if row is None:
            row = {"source": name, "count": 0, "reached_interview": 0,
                   "hired": 0, "companies": set()}
            by_source[name] = row
        row["count"] += 1
        if _reached_interview(entry):
            row["reached_interview"] += 1
        if entry.get("status") == "hired":
            row["hired"] += 1
        company = entry.get("company")
        if isinstance(company, str) and company.strip():
            row["companies"].add(company.strip())
        if name == "referral":
            referrals.append({
                "id": entry.get("id"),
                "referrer": recorded["referrer"],
                "company": entry.get("company"),
                "title": entry.get("title"),
                "status": entry.get("status"),
                "applied_at": entry.get("applied_at"),
                "reached_interview": _reached_interview(entry),
            })
    rows = []
    for name, row in by_source.items():
        if source is not None and name != source:
            continue
        count = row["count"]
        rows.append({
            "source": name,
            "count": count,
            "reached_interview": row["reached_interview"],
            "interview_rate": (row["reached_interview"] / count
                               if count else None),
            "hired": row["hired"],
            "companies": sorted(row["companies"]),
        })
    rows.sort(key=lambda row: (-row["count"], row["source"]))
    # Newest applied first; rows with no usable date sort last, and rows
    # on the same date keep company/title order.
    def _referral_key(referral):
        when = referral.get("applied_at")
        try:
            parsed = (datetime.date.fromisoformat(when)
                      if isinstance(when, str) else None)
        except ValueError:
            parsed = None
        if parsed is None:
            return (1, 0, (referral["company"] or "").lower(),
                    (referral["title"] or "").lower())
        return (0, -parsed.toordinal(), (referral["company"] or "").lower(),
                (referral["title"] or "").lower())
    referrals.sort(key=_referral_key)
    if source is not None and source != "referral":
        referrals = []
    return {
        "sources": rows,
        "referrals": referrals,
        "counts": {
            "total": total,
            "no_source": no_source,
            "with_source": total - no_source,
        },
    }


def sources_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_row(row):
    rate = row["interview_rate"]
    rate_text = f"{rate:.2f}" if rate is not None else "n/a"
    line = (f"{row['source']}: {row['count']} application"
            + ("s" if row["count"] != 1 else "")
            + f", {row['reached_interview']} reached interview "
              f"(rate {rate_text}, n={row['count']})"
            + f", {row['hired']} hired")
    if row["companies"]:
        line += "\n      companies: " + ", ".join(row["companies"])
    return line


def _fmt_referral(referral):
    reached = ("reached interview" if referral["reached_interview"]
               else "has not reached interview")
    when = referral.get("applied_at") or "date not recorded"
    return (f"{referral['referrer']} → {referral['company']} — "
            f"{referral['title']} [{referral['status']}, applied {when}; "
            f"{reached}]")


def render_sources_markdown(data):
    """Human-readable source table: most-used channel first."""
    rows = data["sources"]
    counts = data["counts"]
    header = (f"Sources — {counts['total']} application"
              + ("s" if counts["total"] != 1 else ""))
    if counts["no_source"]:
        header += (f"; {counts['no_source']} with no source recorded")
    lines = [header]
    if not rows:
        lines.append("No sources recorded yet — record how you found a "
                     "posting with `sources --record --id APP-001 "
                     "--source job-board`, or pass `--source` to `apply`.")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.append(f"  {_fmt_row(row)}")
    referrals = data["referrals"]
    if referrals:
        lines.append("")
        lines.append(f"  Referrals ({len(referrals)}):")
        for referral in referrals:
            lines.append(f"    {_fmt_referral(referral)}")
    return "\n".join(lines) + "\n"
