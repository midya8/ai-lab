"""Gap-closing plan (v0.7).

``autojob plan`` turns the digest's gap leaderboard into work to do, and
adds the other half of the gap story: the skills you already have that no
bullet in your profile evidences.

Two lists, both facts-only:

- evidence: a skill in the profile that postings asked for (matched in
  some logged application) but no experience bullet mentions. The plan
  names the skill, its years/level, which postings asked for it, and the
  most recent role as the suggested place to add a real bullet — a
  suggestion, not a claim. Adding the bullet is your work; the plan
  never writes one for you, and it never invents a bullet.
- learn: a skill postings asked for that the profile does not have at
  all — the digest gap leaderboard, reused verbatim (most-missed first,
  required and preferred counted separately, companies listed). The plan
  says plainly: do not claim it until you have it.

``quick_wins`` is the three items that would help the most logged
applications: an evidence item helps every application that asked for
the skill, a learn item helps every application where it was missing.
Evidence items sort before learn items on a tie — one bullet is cheaper
than learning a skill.

Skill data comes only from tracker entries written by ``apply`` since
v0.6, which records the posting's required/preferred skills and which of
them the profile matched. Entries logged before that carry no skill
lists; the plan says how many it skipped instead of guessing — a plan
built on half the log and labelled as if it covered all of it would be
dishonest.
"""

from __future__ import annotations

import datetime
import json

from .digest import gap_leaderboard, has_skill_data
from .profile import skill_map


def _bullets_mention(profile, skill):
    """True when any experience bullet mentions ``skill``.

    Same convention as ``tailor._bullets_for_skill``: the skill name was
    already detected in the posting by word-boundary matching, so a
    substring hit in a bullet is a real mention, not a false positive
    from a longer word.
    """
    wanted = skill.strip().lower()
    for job in profile.get("experience", []):
        for bullet in job.get("bullets", []):
            if wanted in bullet.lower():
                return True
    return False


def _most_recent_role(profile):
    """The experience role with the latest start date (YYYY-MM).

    Undated roles sort last. Returns ``{"title", "company", "start"}``
    or None when the profile has no experience entries. The plan names
    this role only as the suggested place to add an evidencing bullet.
    """
    roles = list(profile.get("experience", []))

    def sort_key(job):
        start = job.get("start", "")
        if len(start) == 7 and start[4] == "-":
            return (-int(start[:4]), -int(start[5:7]))
        return (0, 0)

    roles.sort(key=sort_key)
    if not roles:
        return None
    job = roles[0]
    return {"title": job.get("title", ""),
            "company": job.get("company", ""),
            "start": job.get("start", "")}


def evidence_list(profile, entries):
    """Skills the profile has that postings asked for but no bullet proves.

    Each row counts, over entries with skill data, how many applications
    asked for the skill (required and preferred separately) and the
    companies that asked for it, plus the skill's years/level from the
    profile and the most recent role as the suggested place for a real
    evidencing bullet. Rows sort by asked count first, then skill name —
    so the top row is the unevidenced skill the most postings want.
    """
    skills = skill_map(profile)
    rows = {}
    for entry in entries:
        if not has_skill_data(entry):
            continue
        company = entry.get("company") or "(unknown company)"
        for kind, matched_field in (("required", "required_matched"),
                                    ("preferred", "preferred_matched")):
            for skill in entry.get(matched_field) or []:
                key = skill.strip().lower()
                if key not in skills:
                    continue
                if _bullets_mention(profile, skill):
                    continue
                row = rows.setdefault(key, {
                    "skill": skill,
                    "years": skills[key].get("years"),
                    "level": skills[key].get("level"),
                    "asked_required": 0,
                    "asked_preferred": 0,
                    "companies": [],
                })
                row[f"asked_{kind}"] += 1
                if company not in row["companies"]:
                    row["companies"].append(company)
    for row in rows.values():
        row["companies"].sort()
        row["asked"] = row["asked_required"] + row["asked_preferred"]
    return sorted(rows.values(),
                  key=lambda r: (-r["asked"], r["skill"].lower()))


def quick_wins(evidence, learn):
    """The three plan items that help the most logged applications.

    An evidence item helps every application that asked for the skill
    (``asked``); a learn item helps every application where it was
    missing (``missing``). Evidence sorts first on a tie: one bullet is
    cheaper than learning a skill.
    """
    items = []
    for row in evidence:
        items.append({"skill": row["skill"], "kind": "evidence",
                      "closes": row["asked"]})
    for row in learn:
        items.append({"skill": row["skill"], "kind": "learn",
                      "closes": row["missing"]})
    items.sort(key=lambda item: (-item["closes"],
                                 0 if item["kind"] == "evidence" else 1,
                                 item["skill"].lower()))
    return items[:3]


def build_plan(profile, entries, today=None):
    """Full gap-closing plan dict for the profile + log (JSON-serialisable)."""
    if today is None:
        today = datetime.date.today()
    with_skills = [e for e in entries if has_skill_data(e)]
    evidence = evidence_list(profile, entries)
    learn = gap_leaderboard(entries)
    return {
        "generated_on": today.isoformat(),
        "n_applications": len(entries),
        "n_with_skill_data": len(with_skills),
        "n_without_skill_data": len(entries) - len(with_skills),
        "suggested_role": _most_recent_role(profile),
        "evidence": evidence,
        "learn": learn,
        "quick_wins": quick_wins(evidence, learn),
    }


def plan_json(plan):
    return json.dumps(plan, indent=2, sort_keys=True) + "\n"


def render_plan_markdown(plan):
    """Human-readable gap-closing plan (text)."""
    n = plan["n_applications"]
    if n == 0:
        return ("Gap-closing plan — no applications logged yet\n"
                "Apply to a posting first (see `autojob apply --help`).\n")
    lines = [f"Gap-closing plan — {plan['generated_on']}",
             f"Applications: {n} with skill data: "
             f"{plan['n_with_skill_data']}  "
             f"(skipped: {plan['n_without_skill_data']})", ""]
    evidence = plan["evidence"]
    learn = plan["learn"]
    if not evidence and not learn:
        lines.append("Nothing to close: every skill the postings asked for "
                     "is in the profile and backed by a bullet.")
        if plan["n_without_skill_data"]:
            lines.append(f"  (Note: {plan['n_without_skill_data']} "
                         "application(s) logged before v0.6 carry no skill "
                         "data and are not counted above.)")
        return "\n".join(lines) + "\n"
    wins = plan["quick_wins"]
    if wins:
        lines.append("Quick wins — the fixes that help the most logged "
                     "applications:")
        for i, item in enumerate(wins, 1):
            verb = ("add an evidencing bullet for"
                    if item["kind"] == "evidence" else "learn")
            lines.append(f"  {i}. {verb} {item['skill']} "
                         f"— helps {item['closes']} application(s)")
        lines.append("")
    if evidence:
        lines.append("1. Skills you have but cannot prove yet — add a real "
                     "evidencing bullet (never invent one):")
        role = plan["suggested_role"]
        suggestion = ""
        if role:
            suggestion = (f"  (Suggested place: your most recent role, "
                          f"{role['title']} at {role['company']}.)")
            lines.append(suggestion)
        for i, row in enumerate(evidence, 1):
            years = row.get("years")
            label = row["skill"]
            if years is not None:
                label += f" ({years} yrs"
                label += f", {row['level']}" if row.get("level") else ""
                label += ")"
            parts = []
            if row["asked_required"]:
                parts.append(f"required in {row['asked_required']}")
            if row["asked_preferred"]:
                parts.append(f"preferred in {row['asked_preferred']}")
            lines.append(f"  {i}. {label} — asked in {row['asked']} "
                         f"application(s) ({'; '.join(parts)}); "
                         f"asked by: {', '.join(row['companies'])}")
        lines.append("")
    else:
        lines.append("1. Skills you have but cannot prove yet: none — every "
                     "skill the postings asked for that you have is backed "
                     "by a bullet.")
        lines.append("")
    learn = plan["learn"]
    if learn:
        lines.append("2. Skills to learn — do not claim these until you "
                     "have them:")
        for i, row in enumerate(learn, 1):
            parts = []
            if row["missing_required"]:
                parts.append(f"required in {row['missing_required']} of "
                             f"{row['asked_required']}")
            if row["missing_preferred"]:
                parts.append(f"preferred in {row['missing_preferred']} of "
                             f"{row['asked_preferred']}")
            lines.append(f"  {i}. {row['skill']} — missing in "
                         + "; ".join(parts)
                         + f"; asked by: {', '.join(row['companies'])}")
    else:
        lines.append("2. Skills to learn: none — every skill the postings "
                     "asked for is in your profile.")
    if plan["n_without_skill_data"]:
        lines.append(f"  (Note: {plan['n_without_skill_data']} application(s) "
                     "logged before v0.6 carry no skill data and are not "
                     "counted above.)")
    return "\n".join(lines) + "\n"
