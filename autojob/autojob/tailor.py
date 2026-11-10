"""Tailored resume section builder (v0.5).

`autojob tailor` turns a profile + posting + match into the facts a resume
needs for *this* posting — a skills matrix, the order to list skills in,
and the experience roles ranked by how much of the posting they evidence.

Nothing here writes a resume for you, and nothing is invented:

- The skills matrix covers every skill the posting asks for. A skill the
  profile has shows its years/level and the verbatim bullets that mention
  it; a skill it does not have is listed as a gap, never as a claim.
- `skills_to_lead_with` is the matched skills sorted by years of
  experience (most first), so the strongest evidence goes first. Ties
  break required-before-preferred, then alphabetically — the order is
  deterministic and comes only from the profile.
- `role_ranking` ranks experience roles by how many distinct matched
  posting skills their bullets mention, most recent first on ties. A role
  whose bullets mention no posting skill ranks last and is flagged as
  unevidenced for this posting — the render says to keep it only if it
  shows transferable work, or leave it out. Roles are never reworded and
  bullets are never paraphrased.
- `summary_ingredients` is the profile headline verbatim plus the top
  lead skills with years. It is a list of ingredients for a summary you
  write yourself, not a written summary — autojob does not write claims
  about you in your voice.
"""

from __future__ import annotations

import json

from .match import match_profile
from .profile import skill_map

_BULLETS_PER_SKILL = 2
_BULLETS_PER_ROLE = 3


def _bullets_for_skill(profile, skill):
    """Verbatim bullets mentioning ``skill`` (case-insensitive substring).

    Same convention as ``prep._bullets_for_skill``: the skill name was
    already detected in the posting by word-boundary matching, so a
    substring hit in a bullet is a real mention, not a false positive
    from a longer word.
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


def _matched_skills_in_text(text, matched):
    """Distinct matched skills mentioned in ``text`` (substring, lowercase)."""
    lowered = text.lower()
    return [skill for skill in matched if skill.lower() in lowered]


def _role_ranking(profile, matched):
    """Experience roles ranked by evidenced posting skills, most first.

    Rank key: number of distinct matched posting skills mentioned across
    the role's bullets (descending), then start date (most recent first).
    Roles with no evidenced skill rank last with ``matched_count`` 0.
    """
    ranked = []
    for job in profile.get("experience", []):
        bullets = list(job.get("bullets", []))
        seen_skills = []
        for bullet in bullets:
            for skill in _matched_skills_in_text(bullet, matched):
                if skill not in seen_skills:
                    seen_skills.append(skill)
        evidenced = []
        for bullet in bullets:
            if _matched_skills_in_text(bullet, matched) and len(evidenced) < _BULLETS_PER_ROLE:
                if bullet not in evidenced:
                    evidenced.append(bullet)
        ranked.append({
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "start": job.get("start", ""),
            "end": job.get("end", ""),
            "matched_skills": seen_skills,
            "matched_count": len(seen_skills),
            "bullets": evidenced,
        })

    def sort_key(entry):
        # Most evidenced first; ties break most-recent first, undated last.
        start = entry["start"]
        if len(start) == 7 and start[4] == "-":
            return (-entry["matched_count"], -int(start[:4]), -int(start[5:7]))
        return (-entry["matched_count"], 0, 0)

    ranked.sort(key=sort_key)
    return ranked


def build_tailor(profile, posting, match=None):
    """Build the tailored resume section. Returns a tailor dict."""
    if match is None:
        match = match_profile(profile, posting)
    skills = skill_map(profile)
    matched = match["required_matched"] + match["preferred_matched"]

    matrix = []
    seen = set()
    for skill in match["required_matched"] + match["required_missing"]:
        key = skill.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        matrix.append(_matrix_row(profile, skills, skill, "required"))
    for skill in match["preferred_matched"] + match["preferred_missing"]:
        key = skill.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        matrix.append(_matrix_row(profile, skills, skill, "preferred"))

    lead_with = []
    for row in matrix:
        if row["have"]:
            lead_with.append(row)
    lead_with.sort(key=lambda row: (
        -(row["years"] if row["years"] is not None else -1),
        0 if row["kind"] == "required" else 1,
        row["skill"].lower(),
    ))

    gaps = [row["skill"] for row in matrix if not row["have"]]

    return {
        "role": {
            "title": posting.get("title"),
            "company": posting.get("company"),
            "location": posting.get("location"),
            "remote": posting.get("remote"),
        },
        "match": {
            "score": match["score"],
            "verdict": match["verdict"],
            "required_coverage": match["required_coverage"],
        },
        "skills_matrix": matrix,
        "skills_to_lead_with": [row["skill"] for row in lead_with],
        "role_ranking": _role_ranking(profile, matched),
        "summary_ingredients": {
            "headline": profile.get("headline"),
            "lead_skills": [
                {"skill": row["skill"], "years": row["years"]}
                for row in lead_with[:3]
            ],
        },
        "education": [
            {"degree": item.get("degree", ""),
             "school": item.get("school", ""),
             "year": item.get("year")}
            for item in profile.get("education", [])
        ],
        "not_to_claim": gaps,
    }


def _matrix_row(profile, skills, skill, kind):
    key = skill.strip().lower()
    have = key in skills
    entry = skills.get(key, {})
    return {
        "skill": skill,
        "kind": kind,
        "have": have,
        "years": entry.get("years") if have else None,
        "level": entry.get("level") if have else None,
        "bullets": _bullets_for_skill(profile, skill) if have else [],
    }


def render_tailor_markdown(tailor):
    """Render the tailored resume section as Markdown. Facts only."""
    role = tailor["role"]
    title = role.get("title") or "the open role"
    company = role.get("company") or "the company"
    lines = [f"# Tailored resume — {title} at {company}", ""]

    lines.append("Lead with the skills this posting asks for and you have, "
                 "in this order. Everything below comes from your profile — "
                 "nothing here is a claim about a skill you do not have.")
    lines.append("")

    match = tailor["match"]
    lines.append(f"- Match: {match['verdict']}"
                 + (f" (score {match['score']:.2f})" if match["score"] is not None
                    else " (no score — posting lists no detectable skills)"))
    lines.append("")

    lines.append("## Skills to lead with")
    lines.append("")
    if not tailor["skills_to_lead_with"]:
        lines.append("No matched skills — the posting lists no skills the "
                    "profile has. Lead with your headline and experience "
                    "instead.")
    else:
        for skill in tailor["skills_to_lead_with"]:
            row = next(r for r in tailor["skills_matrix"] if r["skill"] == skill)
            label = skill
            if row.get("years") is not None:
                label += f" ({row['years']} yrs"
                label += f", {row['level']}" if row.get("level") else ""
                label += ")"
            lines.append(f"- {label} [{row['kind']}]")
    lines.append("")

    lines.append("## Skills matrix — every skill the posting asks for")
    lines.append("")
    if not tailor["skills_matrix"]:
        lines.append("The posting lists no detectable skills, so there is "
                    "nothing to match against.")
    for row in tailor["skills_matrix"]:
        if row["have"]:
            label = f"### {row['skill']} [{row['kind']}] — have it"
            if row.get("years") is not None:
                label += f" ({row['years']} yrs"
                label += f", {row['level']}" if row.get("level") else ""
                label += ")"
            lines.append(label)
            lines.append("")
            if row["bullets"]:
                for item in row["bullets"]:
                    lines.append(f"- \"{item['bullet']}\" "
                                 f"({item['title']}, {item['company']})")
            else:
                lines.append("No bullet in the profile mentions this skill. "
                            "It is in your skills list, so you may list it — "
                            "but add a real evidencing bullet to the profile "
                            "first, do not invent one.")
        else:
            lines.append(f"### {row['skill']} [{row['kind']}] — GAP: do not claim")
            lines.append("")
            lines.append("Not in your profile. Do not put this on the resume "
                        "for this posting.")
        lines.append("")

    lines.append("## Experience — most relevant first")
    lines.append("")
    for job in tailor["role_ranking"]:
        lines.append(f"### {job['title']} — {job['company']} "
                     f"({job['start']} to {job['end']})")
        lines.append("")
        if job["matched_count"]:
            lines.append(f"Evidences {job['matched_count']} posting skill(s): "
                         f"{', '.join(job['matched_skills'])}.")
            for bullet in job["bullets"]:
                lines.append(f"- {bullet}")
        else:
            lines.append("No bullet here mentions a skill this posting asks "
                        "for. Keep this role only if it shows transferable "
                        "work; otherwise leave it out of the tailored resume.")
        lines.append("")

    lines.append("## Summary ingredients — write the summary yourself")
    lines.append("")
    ingredients = tailor["summary_ingredients"]
    lines.append(f"- Headline (verbatim): {ingredients['headline']}")
    for item in ingredients["lead_skills"]:
        years = item.get("years")
        lines.append(f"- Lead skill: {item['skill']}"
                     + (f" ({years} yrs)" if years is not None else ""))
    if not ingredients["lead_skills"]:
        lines.append("- No lead skills: the posting lists no skills the "
                    "profile has.")
    lines.append("")

    if tailor["education"]:
        lines.append("## Education")
        lines.append("")
        for item in tailor["education"]:
            year = f" ({item['year']})" if item.get("year") else ""
            lines.append(f"- {item['degree']}, {item['school']}{year}")
        lines.append("")

    lines.append("## Do not claim")
    lines.append("")
    if not tailor["not_to_claim"]:
        lines.append("Nothing missing: every skill the posting asks for is in "
                    "your profile.")
    else:
        for skill in tailor["not_to_claim"]:
            lines.append(f"- {skill}")
    lines.append("")
    return "\n".join(lines)


def tailor_json(tailor):
    return json.dumps(tailor, indent=2, sort_keys=True) + "\n"
