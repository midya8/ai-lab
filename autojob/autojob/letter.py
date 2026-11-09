"""Cover-letter drafting from profile facts only.

The draft is assembled from three sources and nothing else:

1. The profile's own fields (name, headline, summary, contact).
2. Experience bullets quoted verbatim from the profile.
3. The list of skills the profile actually has that the posting asks for.

A skill the profile does not have can never appear in the letter: the
drafting code only ever iterates over matched skills. Anything missing goes
into the review notes, clearly marked "do not claim". The letter is a draft
for a human to review — autojob never sends anything.
"""

from __future__ import annotations

import datetime

from .match import evidence_bullets
from .profile import skill_map, total_experience_years


def draft_letter(profile, posting, match, today=None):
    """Build a cover-letter draft. Returns the letter text.

    ``today`` is a ``datetime.date`` for testing; defaults to today.
    """
    if today is None:
        today = datetime.date.today()
    skills = skill_map(profile)
    title = posting.get("title") or "the open role"
    company = posting.get("company") or "your team"
    name = profile["name"]
    prefs = profile.get("preferences") or {}

    lines = []
    lines.append(name)
    contact = [profile.get("headline", "")]
    if profile.get("location"):
        contact.append(profile["location"])
    if profile.get("email"):
        contact.append(profile["email"])
    if profile.get("phone"):
        contact.append(profile["phone"])
    lines.append(" | ".join(c for c in contact if c))
    lines.append("")
    lines.append(today.strftime("%B %d, %Y"))
    lines.append("")
    lines.append(f"Hiring team, {company}")
    lines.append("")
    lines.append(f"Dear hiring team,")
    lines.append("")
    years = total_experience_years(profile)
    intro = (
        f"I am applying for the {title} role at {company}. "
        f"I am a {profile['headline']} with {years} years of experience"
    )
    if profile.get("summary"):
        intro += f". {profile['summary']}"
    else:
        intro += "."
    lines.append(intro)
    lines.append("")

    bullets = evidence_bullets(profile, match["required_matched"] + match["preferred_matched"])
    if bullets:
        lines.append("Recent work relevant to this role:")
        lines.append("")
        for item in bullets:
            lines.append(f"- {item['bullet']} ({item['title']}, {item['company']})")
        lines.append("")

    matched = match["required_matched"] + [
        s for s in match["preferred_matched"] if s not in match["required_matched"]
    ]
    if matched:
        labelled = []
        for skill in matched:
            entry = skills[skill.strip().lower()]
            years = entry.get("years")
            labelled.append(f"{skill} ({years} yrs)" if years else skill)
        lines.append(
            "Skills I would bring to this role: " + ", ".join(labelled) + "."
        )
        lines.append("")

    remote = posting.get("remote")
    pref_remote = prefs.get("remote")
    if remote == "remote" and pref_remote == "remote-only":
        lines.append("I work remotely and am set up for fully remote collaboration.")
        lines.append("")
    elif remote == "hybrid" and pref_remote in ("remote-only", "hybrid-ok"):
        lines.append("I am open to a hybrid arrangement.")
        lines.append("")
    if prefs.get("notice_weeks") is not None:
        lines.append(
            f"My notice period is {prefs['notice_weeks']} weeks."
        )
        lines.append("")

    lines.append(
        "Thank you for considering my application. I would welcome the chance "
        "to discuss how my experience fits this role."
    )
    lines.append("")
    lines.append("Best regards,")
    lines.append(name)
    return "\n".join(lines) + "\n"


def review_notes(profile, posting, match):
    """Reviewer notes for a packet: gaps, missing info, and next steps.

    This is the honest half of the packet. It lists every required skill the
    profile lacks (do not claim these), any salary-expectation mismatch, and
    the manual steps still needed before submitting.
    """
    lines = ["# Review notes — do not send", ""]
    lines.append("## Match summary")
    lines.append("")
    score = match["score"]
    lines.append(f"- Verdict: **{match['verdict']}**"
                 + (f" (score {score:.2f})" if score is not None else " (no score — posting lists no detectable skills)"))
    lines.append(f"- Required coverage: {_pct(match['required_coverage'])}")
    lines.append(f"- Preferred coverage: {_pct(match['preferred_coverage'])}")
    lines.append(f"- Experience fit: {_pct(match['experience_fit'])}")
    lines.append("")

    lines.append("## Do not claim")
    lines.append("")
    if match["required_missing"]:
        lines.append("Required by the posting, NOT in the profile — leave these out "
                    "of the letter and the application:")
        lines.append("")
        for skill in match["required_missing"]:
            lines.append(f"- {skill}")
    else:
        lines.append("No required skills are missing from the profile.")
    if match["preferred_missing"]:
        lines.append("")
        lines.append("Preferred (nice-to-have) skills not in the profile:")
        lines.append("")
        for skill in match["preferred_missing"]:
            lines.append(f"- {skill}")
    lines.append("")

    bullets = evidence_bullets(profile, match["required_matched"] + match["preferred_matched"])
    if not bullets and (match["required_matched"] or match["preferred_matched"]):
        lines.append("## Evidence gap")
        lines.append("")
        lines.append("No experience bullet in the profile mentions the matched skills. "
                    "The letter therefore cites no specific project. Add a real bullet "
                    "to the profile first, or write one manually — do not invent one.")
        lines.append("")

    salary = posting.get("salary")
    prefs = profile.get("preferences") or {}
    if salary and (prefs.get("salary_min") is not None or prefs.get("salary_max") is not None):
        lines.append("## Salary")
        lines.append("")
        lines.append(f"- Posting range: {salary['raw']}")
        if prefs.get("salary_min") is not None:
            lines.append(f"- Your stated minimum: {prefs['salary_min']}")
        if prefs.get("salary_max") is not None:
            lines.append(f"- Your stated maximum: {prefs['salary_max']}")
        if prefs.get("salary_min") is not None and salary["max"] < prefs["salary_min"]:
            lines.append("- The posting's top is below your stated minimum — check before applying.")
        lines.append("")

    lines.append("## Before submitting")
    lines.append("")
    lines.append("- [ ] Read the draft letter and fix anything that overstates your experience.")
    lines.append("- [ ] Attach your actual resume (autojob never edits it).")
    lines.append("- [ ] Submit the application yourself at the posting's link — autojob does not submit.")
    lines.append("- [ ] Record the outcome with `autojob status` once you hear back.")
    return "\n".join(lines) + "\n"


def _pct(value):
    return f"{value:.0%}" if value is not None else "n/a"
