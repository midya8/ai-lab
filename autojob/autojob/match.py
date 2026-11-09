"""Match scoring between a profile and a parsed posting.

The score is a transparent weighted average of three components, each
reported separately so the number can be checked by hand:

- required coverage (weight 0.6): required skills the profile has / required
- preferred coverage (weight 0.25): preferred skills the profile has / preferred
- experience fit (weight 0.15): required skills backed by >= 1 year in the profile

A component with nothing to measure (no required skills in the posting) is
left out of the average instead of being scored as 1.0 — claiming full
coverage of nothing would be dishonest. Gaps are listed explicitly: they are
the skills the letter must NOT claim.
"""

from __future__ import annotations

from .profile import skill_map

WEIGHTS = {"required": 0.6, "preferred": 0.25, "experience": 0.15}


def match_profile(profile, posting):
    """Score a profile against a parsed posting. Returns a match dict."""
    skills = skill_map(profile)
    required = posting.get("required_skills") or []
    preferred = posting.get("preferred_skills") or []

    def has(skill):
        return skill.strip().lower() in skills

    def experienced(skill):
        entry = skills.get(skill.strip().lower())
        return bool(entry) and (entry.get("years") or 0) >= 1

    req_matched = [s for s in required if has(s)]
    req_missing = [s for s in required if not has(s)]
    pref_matched = [s for s in preferred if has(s)]
    pref_missing = [s for s in preferred if not has(s)]

    req_cov = (len(req_matched) / len(required)) if required else None
    pref_cov = (len(pref_matched) / len(preferred)) if preferred else None
    exp_fit = None
    if required:
        exp_fit = sum(1 for s in required if experienced(s)) / len(required)

    parts = []
    if req_cov is not None:
        parts.append((req_cov, WEIGHTS["required"]))
    if pref_cov is not None:
        parts.append((pref_cov, WEIGHTS["preferred"]))
    if exp_fit is not None:
        parts.append((exp_fit, WEIGHTS["experience"]))
    if parts:
        score = sum(value * weight for value, weight in parts) / sum(
            weight for _, weight in parts
        )
    else:
        score = None

    if score is None:
        verdict = "unknown"
    elif (req_cov or 0) >= 0.8:
        verdict = "strong"
    elif (req_cov or 0) >= 0.5:
        verdict = "partial"
    else:
        verdict = "weak"

    return {
        "score": round(score, 3) if score is not None else None,
        "verdict": verdict,
        "required_coverage": round(req_cov, 3) if req_cov is not None else None,
        "preferred_coverage": round(pref_cov, 3) if pref_cov is not None else None,
        "experience_fit": round(exp_fit, 3) if exp_fit is not None else None,
        "required_matched": req_matched,
        "required_missing": req_missing,
        "preferred_matched": pref_matched,
        "preferred_missing": pref_missing,
        "weights": dict(WEIGHTS),
    }


def evidence_bullets(profile, matched_skills, limit=3):
    """Verbatim experience bullets that evidence the matched skills.

    Each bullet is returned with its role and company so the letter can
    attribute it. Bullets are quoted exactly as written in the profile —
    never paraphrased into a stronger claim. Returns [] when no bullet
    mentions any matched skill; the caller must say so in the review notes
    instead of inventing evidence.
    """
    wanted = [s.lower() for s in matched_skills]
    picked = []
    for job in profile.get("experience", []):
        for bullet in job.get("bullets", []):
            lowered = bullet.lower()
            if any(skill in lowered for skill in wanted):
                entry = {
                    "bullet": bullet,
                    "title": job.get("title", ""),
                    "company": job.get("company", ""),
                }
                if entry not in picked:
                    picked.append(entry)
                if len(picked) >= limit:
                    return picked
    return picked
