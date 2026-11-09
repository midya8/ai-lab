"""Job-posting parser.

Takes the raw text of a job posting (Markdown or plain text) and extracts a
structured summary: title, company, location/remote signals, salary range,
and required vs preferred skills. Skill detection is keyword matching
against a vocabulary built from the profile's own skills plus a built-in
list of common data/BI/engineering skills — it finds what the posting asks
for, it never invents requirements.
"""

from __future__ import annotations

import re

# Common skills for data / BI / analytics / engineering roles. The profile's
# own skill names are always added to this vocabulary at parse time, so a
# profile skill is detectable even if it is not in this list.
COMMON_SKILLS = [
    "SQL", "Python", "R", "Excel", "Power BI", "Tableau", "Looker", "dbt",
    "Snowflake", "BigQuery", "Redshift", "PostgreSQL", "MySQL", "Spark",
    "Airflow", "Git", "Docker", "Kubernetes", "AWS", "Azure", "GCP",
    "JavaScript", "TypeScript", "React", "Node.js", "Java", "Go", "Rust",
    "Pandas", "NumPy", "scikit-learn", "TensorFlow", "PyTorch", "Machine Learning",
    "Statistics", "A/B Testing", "ETL", "Data Modeling", "Data Warehousing",
    "APIs", "REST", "GraphQL", "Linux", "Jira", "Confluence", "Figma",
    "Salesforce", "HubSpot", "GA4", "Google Analytics", "Amplitude", "Mixpanel",
    "Kafka", "dbt Cloud", "Sigma", "Mode", "Metabase", "Superset",
    "CI/CD", "Terraform", "Agile", "Scrum", "Stakeholder Management",
    "Communication", "Project Management", "Data Governance", "dbt Semantic Layer",
]

REQUIRED_HEADINGS = ("requirement", "qualifications", "must have", "what you bring",
                     "what you'll bring", "required", "you have", "you'll have")
PREFERRED_HEADINGS = ("nice to have", "preferred", "bonus", "plus", "nice-to-have",
                      "great to have", "desirable")

_SALARY_RE = re.compile(
    r"(?P<cur>USD|CAD|GBP|EUR|AUD|\$|£|€)?\s*"
    r"(?P<lo>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<lok>[kK])?"
    r"\s*(?:-|–|—|to)\s*"
    r"(?P<cur2>USD|CAD|GBP|EUR|AUD|\$|£|€)?\s*"
    r"(?P<hi>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<hik>[kK])?"
)
_SINGLE_SALARY_RE = re.compile(
    r"(?P<cur>USD|CAD|GBP|EUR|AUD|\$|£|€)\s*"
    r"(?P<val>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?P<k>[kK])?"
)


def _number(raw, k_marker):
    value = float(raw.replace(",", ""))
    if k_marker:
        value *= 1000
    return value


def parse_salary(text):
    """Parse the first salary range in the text, or None.

    Returns ``{"min": ..., "max": ..., "currency": ..., "raw": ...}``.
    A lone figure (``$110,000``) is returned as min == max.
    """
    match = _SALARY_RE.search(text)
    if match:
        lo = _number(match.group("lo"), match.group("lok"))
        hi = _number(match.group("hi"), match.group("hik"))
        cur = match.group("cur") or match.group("cur2") or ""
        return {"min": lo, "max": hi, "currency": cur or None, "raw": match.group(0).strip()}
    single = _SINGLE_SALARY_RE.search(text)
    if single:
        value = _number(single.group("val"), single.group("k"))
        return {"min": value, "max": value,
                "currency": single.group("cur") or None,
                "raw": single.group(0).strip()}
    return None


def _split_sections(lines):
    """Split posting lines into (section_name, text) by Markdown/plan headings."""
    sections = []
    current_name = "intro"
    current_lines = []
    for line in lines:
        stripped = line.strip()
        heading = re.match(r"^#{1,4}\s+(.*)$", stripped)
        plain = re.match(r"^([A-Z][A-Za-z /&-]{3,60}):\s*$", stripped)
        label = None
        if heading:
            label = heading.group(1).strip().lower()
        elif plain:
            label = plain.group(1).strip().lower()
        if label is not None:
            sections.append((current_name, "\n".join(current_lines)))
            current_name = label
            current_lines = []
        else:
            current_lines.append(line)
    sections.append((current_name, "\n".join(current_lines)))
    return sections


def _section_kind(name):
    lowered = name.lower()
    for marker in PREFERRED_HEADINGS:
        if marker in lowered:
            return "preferred"
    for marker in REQUIRED_HEADINGS:
        if marker in lowered:
            return "required"
    return None


def find_skills(text, vocabulary):
    """Return the vocabulary skills mentioned in ``text`` (word-boundary match).

    Multi-word skills (``Power BI``) are matched with flexible whitespace.
    Single letters (``R``) only match as a standalone word, never inside
    another word.
    """
    found = []
    for skill in vocabulary:
        pattern = r"\b" + re.escape(skill).replace(r"\ ", r"\s+") + r"\b"
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(skill)
    return found


def parse_posting(text, profile_skills=()):
    """Parse a posting into a structured dict."""
    lines = text.splitlines()
    title = None
    company = None
    for line in lines:
        stripped = line.strip()
        if not title:
            m = re.match(r"^#\s+(.+)$", stripped)
            if m:
                title = m.group(1).strip()
                # "Data Analyst at Northwind Analytics" — split the company
                # out of the title when the tail looks like a company name.
                if " at " in title:
                    head, _, tail = title.rpartition(" at ")
                    if " " in tail.strip():
                        title, company = head.strip(), tail.strip()
                continue
            m = re.match(r"^(?:title|role|position)\s*:\s*(.+)$", stripped, re.IGNORECASE)
            if m:
                title = m.group(1).strip()
                continue
        if not company:
            m = re.match(r"^(?:company|employer|organization)\s*:\s*(.+)$",
                         stripped, re.IGNORECASE)
            if m:
                company = m.group(1).strip()
                continue
        if title and company:
            break
    if title and " at " in title and not company:
        # "Data Analyst at Northwind" — split the company out of the title.
        head, _, tail = title.rpartition(" at ")
        if " " in tail.strip():
            title, company = head.strip(), tail.strip()

    location = None
    m = re.search(r"(?:location|based in|office)\s*:\s*(.+)$", text,
                  flags=re.IGNORECASE | re.MULTILINE)
    if m:
        location = m.group(1).strip()

    lowered = text.lower()
    remote = None
    if re.search(r"\bfully remote\b|\bremote-first\b|\bremote only\b|\b100% remote\b", lowered):
        remote = "remote"
    elif re.search(r"\bhybrid\b", lowered):
        remote = "hybrid"
    elif re.search(r"\bon-site\b|\bonsite\b|\bin-office\b", lowered):
        remote = "onsite"
    elif re.search(r"\bremote\b", lowered):
        remote = "remote"

    vocabulary = list(dict.fromkeys(list(COMMON_SKILLS) + [s for s in profile_skills]))
    sections = _split_sections(lines)
    required, preferred = [], []
    seen_required, seen_preferred = set(), set()
    for name, body in sections:
        kind = _section_kind(name)
        found = find_skills(body, vocabulary)
        if kind == "preferred":
            for skill in found:
                if skill.lower() not in seen_preferred:
                    preferred.append(skill)
                    seen_preferred.add(skill.lower())
        else:
            # Unlabelled sections and required sections both count as required;
            # a skill found anywhere outside a "preferred" section is required.
            for skill in found:
                if skill.lower() not in seen_required:
                    required.append(skill)
                    seen_required.add(skill.lower())
    # A skill listed as both required and preferred stays required only.
    preferred = [s for s in preferred if s.lower() not in seen_required]

    return {
        "title": title,
        "company": company,
        "location": location,
        "remote": remote,
        "salary": parse_salary(text),
        "required_skills": required,
        "preferred_skills": preferred,
    }
