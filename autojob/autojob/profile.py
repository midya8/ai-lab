"""Profile loading and validation.

A profile is a JSON file describing one real person: name, headline, contact,
skills (with years), experience (with verbatim bullets), education, and job
preferences. Validation is strict about shape and honest about content: it
checks that the file is well-formed, but it cannot verify that the facts are
true — that is the user's responsibility. Nothing here invents data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_DATE_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

REQUIRED_TOP = ("name", "headline", "skills", "experience")
OPTIONAL_TOP = ("location", "email", "phone", "summary", "education", "preferences")
PREFERENCE_KEYS = ("remote", "notice_weeks", "salary_min", "salary_max", "currency")
REMOTE_VALUES = ("remote-only", "hybrid-ok", "onsite-ok", "any")


class ProfileError(ValueError):
    """A profile file failed validation; ``errors`` lists every problem found."""


def load_profile(path):
    """Load and validate a profile JSON file. Returns the profile dict.

    Raises ProfileError with a list of ``path: message`` problems.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"profile: invalid JSON: {exc}") from exc
    errors = validate_profile(data)
    if errors:
        raise ProfileError("; ".join(errors))
    return data


def validate_profile(data):
    """Return a list of ``path: message`` problems; empty means valid."""
    errors = []
    if not isinstance(data, dict):
        return ["profile: must be a JSON object"]
    for key in REQUIRED_TOP:
        if key not in data:
            errors.append(f"{key}: missing required field")
    if not isinstance(data.get("name", ""), str) or not data.get("name", "").strip():
        errors.append("name: must be a non-empty string")
    if not isinstance(data.get("headline", ""), str) or not data.get("headline", "").strip():
        errors.append("headline: must be a non-empty string")
    if "email" in data and data["email"] is not None:
        if not isinstance(data["email"], str) or not _EMAIL_RE.match(data["email"]):
            errors.append("email: must look like name@example.com")
    if "phone" in data and data["phone"] is not None:
        if not isinstance(data["phone"], str) or not data["phone"].strip():
            errors.append("phone: must be a non-empty string")
    if "summary" in data and data["summary"] is not None:
        if not isinstance(data["summary"], str) or not data["summary"].strip():
            errors.append("summary: must be a non-empty string")
    errors.extend(_validate_skills(data.get("skills")))
    errors.extend(_validate_experience(data.get("experience")))
    errors.extend(_validate_education(data.get("education")))
    errors.extend(_validate_preferences(data.get("preferences")))
    return errors


def _validate_skills(skills):
    errors = []
    if skills is None:
        return errors  # already reported as missing
    if not isinstance(skills, list) or not skills:
        return ["skills: must be a non-empty list"]
    seen = set()
    for i, skill in enumerate(skills):
        base = f"skills[{i}]"
        if not isinstance(skill, dict):
            errors.append(f"{base}: must be an object")
            continue
        name = skill.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{base}.name: must be a non-empty string")
        else:
            key = name.strip().lower()
            if key in seen:
                errors.append(f"{base}.name: duplicate skill {name!r}")
            seen.add(key)
        years = skill.get("years")
        if years is not None and (not isinstance(years, (int, float)) or years < 0):
            errors.append(f"{base}.years: must be a non-negative number")
        level = skill.get("level")
        if level is not None and level not in ("beginner", "intermediate", "advanced", "expert"):
            errors.append(f"{base}.level: must be beginner/intermediate/advanced/expert")
    return errors


def _validate_experience(experience):
    errors = []
    if experience is None:
        return errors
    if not isinstance(experience, list):
        return ["experience: must be a list"]
    for i, job in enumerate(experience):
        base = f"experience[{i}]"
        if not isinstance(job, dict):
            errors.append(f"{base}: must be an object")
            continue
        for field in ("title", "company", "start"):
            if not isinstance(job.get(field), str) or not job.get(field, "").strip():
                errors.append(f"{base}.{field}: must be a non-empty string")
        if isinstance(job.get("start"), str) and not _DATE_RE.match(job["start"]):
            errors.append(f"{base}.start: must be YYYY-MM")
        end = job.get("end")
        if end is not None and end != "present":
            if not isinstance(end, str) or not _DATE_RE.match(end):
                errors.append(f"{base}.end: must be YYYY-MM or 'present'")
        bullets = job.get("bullets")
        if bullets is not None:
            if not isinstance(bullets, list) or any(
                not isinstance(b, str) or not b.strip() for b in bullets
            ):
                errors.append(f"{base}.bullets: must be a list of non-empty strings")
    return errors


def _validate_education(education):
    errors = []
    if education is None:
        return errors
    if not isinstance(education, list):
        return ["education: must be a list"]
    for i, item in enumerate(education):
        base = f"education[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{base}: must be an object")
            continue
        for field in ("degree", "school"):
            if not isinstance(item.get(field), str) or not item.get(field, "").strip():
                errors.append(f"{base}.{field}: must be a non-empty string")
        year = item.get("year")
        if year is not None and (not isinstance(year, int) or year < 1900 or year > 2100):
            errors.append(f"{base}.year: must be a plausible year")
    return errors


def _validate_preferences(prefs):
    errors = []
    if prefs is None:
        return errors
    if not isinstance(prefs, dict):
        return ["preferences: must be an object"]
    for key in prefs:
        if key not in PREFERENCE_KEYS:
            errors.append(f"preferences.{key}: unknown preference")
    remote = prefs.get("remote")
    if remote is not None and remote not in REMOTE_VALUES:
        errors.append(f"preferences.remote: must be one of {', '.join(REMOTE_VALUES)}")
    weeks = prefs.get("notice_weeks")
    if weeks is not None and (not isinstance(weeks, int) or weeks < 0):
        errors.append("preferences.notice_weeks: must be a non-negative integer")
    for key in ("salary_min", "salary_max"):
        value = prefs.get(key)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            errors.append(f"preferences.{key}: must be a non-negative number")
    lo, hi = prefs.get("salary_min"), prefs.get("salary_max")
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > hi:
        errors.append("preferences: salary_min must not exceed salary_max")
    return errors


def skill_map(profile):
    """Map normalized skill name -> skill dict from the profile."""
    out = {}
    for skill in profile.get("skills", []):
        out[skill["name"].strip().lower()] = skill
    return out


def total_experience_years(profile, today=None):
    """Total years across experience entries, from YYYY-MM ranges.

    ``today`` is a ``(year, month)`` pair for testing; defaults to the
    current month. Overlapping roles are counted once each (summed), which
    is a rough total — the report says so.
    """
    import datetime

    if today is None:
        now = datetime.date.today()
        today = (now.year, now.month)
    total_months = 0
    for job in profile.get("experience", []):
        start = job.get("start", "")
        end = job.get("end") or "present"
        if not _DATE_RE.match(start):
            continue
        sy, sm = int(start[:4]), int(start[5:7])
        if end == "present":
            ey, em = today
        elif _DATE_RE.match(end):
            ey, em = int(end[:4]), int(end[5:7])
        else:
            continue
        months = (ey - sy) * 12 + (em - sm) + 1
        if months > 0:
            total_months += months
    return round(total_months / 12, 1)
