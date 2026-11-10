"""Interview-prep packet builder (v0.3).

`autojob prep` turns a profile + posting + match into a prep sheet for the
interview — talking points, gaps to prepare for honestly, likely questions,
and a story bank. Every line in it comes from the profile or the posting:

- Talking points pair each matched skill with the profile's years/level and
  the verbatim experience bullets that mention it. A matched skill with no
  bullet is flagged as unevidenced, not dressed up with invented evidence.
- Gaps list the skills the posting asks for that the profile does not have.
  They are preparation prompts ("be ready to answer honestly"), never claims.
- Likely questions are templates filled only with posting/profile facts
  (skill names, role, company, most recent employer). They are questions,
  not statements about the candidate.
- The story bank quotes experience entries verbatim, most recent first.

The packet is a study aid for the candidate. It is not sent anywhere and it
never invents experience.
"""

from __future__ import annotations

import json

from .match import match_profile
from .profile import skill_map, total_experience_years

_BULLETS_PER_SKILL = 2


def _bullets_for_skill(profile, skill):
    """Verbatim bullets mentioning ``skill`` (case-insensitive substring).

    Same convention as ``match.evidence_bullets``: the skill name was already
    detected in the posting by word-boundary matching, so a substring hit in
    a bullet is a real mention, not a false positive from a longer word.
    """
    wanted = skill.lower()
    picked = []
    for job in profile.get("experience", []):
        for bullet in job.get("bullets", []):
            if wanted in bullet.lower():
                entry = {"bullet": bullet,
                         "title": job.get("title", ""),
                         "company": job.get("company", "")}
                if entry not in picked:
                    picked.append(entry)
                if len(picked) >= _BULLETS_PER_SKILL:
                    return picked
    return picked


def _story_bank(profile):
    """Experience entries, most recent first, verbatim.

    Sorted by start date descending (``YYYY-MM`` sorts lexicographically);
    entries without a usable start date keep their profile order at the end.
    """
    jobs = list(profile.get("experience", []))

    def start_key(job):
        start = job.get("start") or ""
        return (1, start) if len(start) == 7 and start[4] == "-" else (0, "")

    ordered = sorted(jobs, key=start_key, reverse=True)
    return [{"title": job.get("title", ""),
             "company": job.get("company", ""),
             "start": job.get("start", ""),
             "end": job.get("end", ""),
             "bullets": list(job.get("bullets", []))} for job in ordered]


def build_prep(profile, posting, match=None):
    """Build the interview-prep packet. Returns a prep dict."""
    if match is None:
        match = match_profile(profile, posting)
    skills = skill_map(profile)
    prefs = profile.get("preferences") or {}

    talking_points = []
    seen = set()
    for skill in match["required_matched"] + match["preferred_matched"]:
        key = skill.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        entry = skills.get(key, {})
        talking_points.append({
            "skill": skill,
            "kind": "required" if skill in match["required_matched"] else "preferred",
            "years": entry.get("years"),
            "level": entry.get("level"),
            "bullets": _bullets_for_skill(profile, skill),
        })

    gap_prep = []
    for skill in match["required_missing"]:
        gap_prep.append({"skill": skill, "kind": "required"})
    for skill in match["preferred_missing"]:
        gap_prep.append({"skill": skill, "kind": "preferred"})

    questions = []
    for skill in match["required_matched"]:
        questions.append(f"Tell me about a time you used {skill} in your work.")
    stories = _story_bank(profile)
    if stories:
        recent = stories[0]
        questions.append(
            f"Walk me through your work as {recent['title']} at {recent['company']}.")
    title = posting.get("title")
    company = posting.get("company")
    if title and company:
        questions.append(f"Why are you interested in the {title} role at {company}?")
    elif title:
        questions.append(f"Why are you interested in the {title} role?")
    if posting.get("salary"):
        questions.append("What are your salary expectations for this role?")

    salary = posting.get("salary")
    return {
        "role": {
            "title": title,
            "company": company,
            "location": posting.get("location"),
            "remote": posting.get("remote"),
            "salary_raw": salary["raw"] if salary else None,
        },
        "match": {
            "score": match["score"],
            "verdict": match["verdict"],
            "required_coverage": match["required_coverage"],
        },
        "talking_points": talking_points,
        "gap_prep": gap_prep,
        "likely_questions": questions,
        "facts": {
            "headline": profile.get("headline"),
            "total_experience_years": total_experience_years(profile),
            "notice_weeks": prefs.get("notice_weeks"),
            "salary_min": prefs.get("salary_min"),
            "salary_max": prefs.get("salary_max"),
            "currency": prefs.get("currency"),
            "remote_preference": prefs.get("remote"),
        },
        "story_bank": stories,
    }


def render_prep_markdown(prep):
    """Render the prep packet as Markdown. Facts only — see module docstring."""
    role = prep["role"]
    title = role.get("title") or "the open role"
    company = role.get("company") or "the company"
    lines = [f"# Interview prep — {title} at {company}", ""]

    lines.append("## The role")
    lines.append("")
    lines.append(f"- Company: {company}")
    lines.append(f"- Role: {title}")
    if role.get("location"):
        lines.append(f"- Location: {role['location']}")
    if role.get("remote"):
        lines.append(f"- Work mode: {role['remote']}")
    if role.get("salary_raw"):
        lines.append(f"- Posting salary: {role['salary_raw']}")
    match = prep["match"]
    lines.append(f"- Match: {match['verdict']}"
                 + (f" (score {match['score']:.2f})" if match["score"] is not None
                    else " (no score — posting lists no detectable skills)"))
    lines.append("")

    lines.append("## Talking points — skills you have, with evidence")
    lines.append("")
    if not prep["talking_points"]:
        lines.append("No matched skills — the posting lists no skills the profile has. "
                    "Prepare from the story bank below instead.")
    for point in prep["talking_points"]:
        years = point.get("years")
        level = point.get("level")
        label = point["skill"]
        if years is not None:
            label += f" ({years} yrs"
            label += f", {level}" if level else ""
            label += ")"
        lines.append(f"### {label} [{point['kind']}]")
        lines.append("")
        if point["bullets"]:
            for item in point["bullets"]:
                lines.append(f"- \"{item['bullet']}\" ({item['title']}, {item['company']})")
        else:
            lines.append("No bullet in the profile mentions this skill. Prepare from "
                        "memory, or add a real bullet to the profile first — "
                        "do not invent one.")
        lines.append("")

    lines.append("## Gaps — prepare an honest answer, do not claim these")
    lines.append("")
    if not prep["gap_prep"]:
        lines.append("No gaps: every skill the posting asks for is in the profile.")
    else:
        for gap in prep["gap_prep"]:
            lines.append(f"- {gap['skill']} [{gap['kind']}] — not in your profile. "
                        "If asked, say so plainly and talk about adjacent "
                        "experience or how you would learn it.")
    lines.append("")

    lines.append("## Likely questions")
    lines.append("")
    for question in prep["likely_questions"]:
        lines.append(f"- {question}")
    lines.append("")

    facts = prep["facts"]
    lines.append("## Your facts — for the negotiation half")
    lines.append("")
    lines.append(f"- Headline: {facts['headline']}")
    lines.append(f"- Total experience: {facts['total_experience_years']} years")
    if facts.get("notice_weeks") is not None:
        lines.append(f"- Notice period: {facts['notice_weeks']} weeks")
    if facts.get("salary_min") is not None or facts.get("salary_max") is not None:
        lines.append(f"- Your stated salary range: {facts['salary_min']}–"
                     f"{facts['salary_max']} {facts.get('currency') or ''}".rstrip())
    if facts.get("remote_preference"):
        lines.append(f"- Work preference: {facts['remote_preference']}")
    lines.append("")

    lines.append("## Story bank — your experience, verbatim, most recent first")
    lines.append("")
    for job in prep["story_bank"]:
        lines.append(f"### {job['title']} — {job['company']} "
                     f"({job['start']} to {job['end']})")
        lines.append("")
        for bullet in job["bullets"]:
            lines.append(f"- {bullet}")
        lines.append("")
    return "\n".join(lines)


def prep_json(prep):
    return json.dumps(prep, indent=2, sort_keys=True) + "\n"
