"""Salary analysis: what postings pay vs what the profile asks for.

``salary`` reads every ``.md`` / ``.txt`` posting in a directory, takes the
salary range each one lists (see posting.parse_salary), and compares it
with the salary target in the profile's preferences (salary_min /
salary_max / currency). The comparison is deliberately narrow, because
the interesting mistakes here are the silent ones:

- Ranges are compared as listed. A posting that does not state a period
  is treated as annual — the assumption is printed on every report —
  and an hourly rate is listed but never compared with an annual target.
- Currencies are never converted. A comparison happens only when the
  posting states the same three-letter currency code as the profile
  target (``CAD`` vs ``CAD``); a bare ``$`` or a different code means
  "not compared", with the reason on the row, never a guessed rate.
- Medians are per currency, over postings that list a range in that
  currency and are not hourly. A posting with no range is listed, not
  dropped, so the report never hides how much is unknown.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .posting import parse_posting
from .rank import POSTING_SUFFIXES, _applied_map
from .tracker import _norm

CURRENCY_CODES = ("CAD", "USD", "GBP", "EUR", "AUD")

_HOURLY_RE = re.compile(r"per\s+hour|/hr\b|hourly", re.IGNORECASE)
_ANNUAL_RE = re.compile(r"per\s+year|/yr\b|annual|per\s+annum|\bp\.a\.", re.IGNORECASE)


def target_from_profile(profile):
    """The profile's salary target, or None when it sets no salary range.

    Returns ``{"min": ..., "max": ..., "currency": ...}`` with either
    bound allowed to be None. A profile that sets only a minimum is a
    floor; one that sets only a maximum is a ceiling.
    """
    prefs = profile.get("preferences") or {}
    lo = prefs.get("salary_min")
    hi = prefs.get("salary_max")
    if lo is None and hi is None:
        return None
    return {"min": lo, "max": hi, "currency": prefs.get("currency")}


def _detect_period(text, salary):
    """annual / hourly / unspecified, from the words near the salary range.

    Only the 60 characters after the matched range are read, so a stray
    "hourly" elsewhere in the posting cannot mislabel the range.
    """
    raw = salary["raw"]
    idx = text.find(raw)
    window = text[idx:idx + len(raw) + 60] if idx >= 0 else ""
    if _HOURLY_RE.search(window):
        return "hourly"
    if _ANNUAL_RE.search(window):
        return "annual"
    return "unspecified"


def _comparable(posting_currency, target_currency):
    """True only when both sides state the same three-letter code."""
    return (posting_currency in CURRENCY_CODES
            and target_currency in CURRENCY_CODES
            and posting_currency == target_currency)


def _verdict(salary, period, target):
    """(verdict, reason) for one posting against the target.

    Verdicts: "meets target", "below target" (its max is under your
    minimum), "above target" (its min is over your maximum), "no range
    listed", "hourly — not compared", "no target set", "not compared".
    A verdict of "not compared" always carries the reason why.
    """
    if salary is None:
        return "no range listed", None
    if period == "hourly":
        return "hourly — not compared", None
    if target is None:
        return "no target set", None
    posting_currency = salary.get("currency")
    target_currency = target.get("currency")
    if not _comparable(posting_currency, target_currency):
        if posting_currency not in CURRENCY_CODES:
            reason = "posting does not state a currency code"
        elif target_currency not in CURRENCY_CODES:
            reason = "profile target has no currency"
        else:
            reason = (f"posting is {posting_currency}, "
                      f"target is {target_currency}")
        return "not compared", reason
    if target.get("min") is not None and salary["max"] < target["min"]:
        return "below target", None
    if target.get("max") is not None and salary["min"] > target["max"]:
        return "above target", None
    return "meets target", None


def _median(values):
    """Median of a non-empty list of numbers; None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def analyze_salaries(profile, postings_dir, entries=(), min_pay=None):
    """Analyse every posting file in ``postings_dir`` against the profile.

    Returns a dict with the target, one row per posting (best-paying
    first, no-range rows last), per-currency medians, verdict counts,
    and skipped files. ``min_pay`` drops rows whose max is below the
    floor — but only rows in the target's currency, which is the only
    currency a floor in the target's terms can judge; rows that cannot
    be judged are kept and labelled. Unreadable files land in
    ``skipped`` and never crash the run.
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
    target = target_from_profile(profile)
    rows, skipped = [], []
    n_scored = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            skipped.append({"file": path.name, "reason": str(exc)})
            continue
        posting = parse_posting(text, profile_skills)
        salary = posting.get("salary")
        period = _detect_period(text, salary) if salary else None
        verdict, reason = _verdict(salary, period, target)
        n_scored += 1
        row = {
            "file": path.name,
            "title": posting.get("title") or "unknown role",
            "company": posting.get("company") or "unknown company",
            "salary": salary,
            "period": period,
            "midpoint": ((salary["min"] + salary["max"]) / 2
                         if salary else None),
            "verdict": verdict,
            "verdict_reason": reason,
            "applied": False,
            "applied_id": None,
        }
        key = (_norm(posting.get("company")), _norm(posting.get("title")))
        applied_id = applied.get(key)
        row["applied"] = applied_id is not None
        row["applied_id"] = applied_id
        if (min_pay is not None and salary is not None
                and period != "hourly"
                and _comparable(salary.get("currency"),
                                (target or {}).get("currency"))
                and salary["max"] < min_pay):
            continue
        rows.append(row)

    def sort_key(row):
        salary = row["salary"]
        return (
            0 if salary is not None else 1,
            -(salary["max"] if salary is not None else 0.0),
            -(salary["min"] if salary is not None else 0.0),
            row["title"].lower(),
            row["file"],
        )

    rows.sort(key=sort_key)

    grouped = {}
    for row in rows:
        salary = row["salary"]
        if salary is None or row["period"] == "hourly":
            continue
        currency = salary.get("currency")
        if currency not in CURRENCY_CODES:
            continue
        group = grouped.setdefault(currency, {"mins": [], "maxes": [],
                                              "mids": []})
        group["mins"].append(salary["min"])
        group["maxes"].append(salary["max"])
        group["mids"].append(row["midpoint"])
    medians = {
        currency: {
            "n": len(group["mins"]),
            "median_min": _median(group["mins"]),
            "median_max": _median(group["maxes"]),
            "median_mid": _median(group["mids"]),
        }
        for currency, group in sorted(grouped.items())
    }

    counts = {
        "with_range": sum(1 for r in rows if r["salary"] is not None),
        "without_range": sum(1 for r in rows if r["salary"] is None),
        "hourly": sum(1 for r in rows if r["period"] == "hourly"),
        "meets_target": sum(1 for r in rows if r["verdict"] == "meets target"),
        "below_target": sum(1 for r in rows if r["verdict"] == "below target"),
        "above_target": sum(1 for r in rows if r["verdict"] == "above target"),
        "not_compared": sum(1 for r in rows if r["verdict"] == "not compared"),
        "applied": sum(1 for r in rows if r["applied"]),
    }
    return {
        "target": target,
        "rows": rows,
        "medians": medians,
        "counts": counts,
        "skipped": skipped,
        "n_scored": n_scored,
        "n_files": len(files),
        "n_skipped": len(skipped),
        "min_pay": min_pay,
    }


def salary_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_money(value):
    if value is None:
        return "—"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _target_text(target):
    if target is None:
        return None
    currency = target.get("currency") or ""
    prefix = f"{currency} " if currency else ""
    lo, hi = target.get("min"), target.get("max")
    if lo is not None and hi is not None:
        return f"{prefix}{_fmt_money(lo)}–{_fmt_money(hi)}"
    if lo is not None:
        return f"at least {prefix}{_fmt_money(lo)}"
    return f"up to {prefix}{_fmt_money(hi)}"


def _range_text(row):
    salary = row["salary"]
    if salary is None:
        return "no range listed"
    currency = salary.get("currency") or ""
    prefix = f"{currency} " if currency else ""
    text = (f"{prefix}{_fmt_money(salary['min'])}–{_fmt_money(salary['max'])}"
            f" (mid {_fmt_money(row['midpoint'])})")
    if row["period"] == "hourly":
        text += " per hour"
    return text


def render_salary_markdown(data):
    """Human-readable salary analysis, best-paying first."""
    rows = data["rows"]
    lines = [f"Salary analysis — {len(rows)} postings "
             f"({data['n_files']} files, {data['n_skipped']} skipped)"]
    if data["min_pay"] is not None:
        lines[0] += f", min pay {data['min_pay']}"
    lines.append("")
    target_text = _target_text(data["target"])
    if target_text is None:
        lines.append("No salary target in the profile — ranges are "
                     "reported, not compared.")
    else:
        lines.append(f"Target: {target_text}")
    lines.append("Ranges are compared as listed. A posting that does not "
                 "state a period is treated as annual; hourly rates are "
                 "listed but never compared. Currencies are never "
                 "converted — a range is compared only when it states "
                 "the same currency code as the target.")
    lines.append("")
    if data["medians"]:
        for currency, med in sorted(data["medians"].items()):
            lines.append(f"{currency} ({med['n']} with a range): median "
                         f"{_fmt_money(med['median_min'])}–"
                         f"{_fmt_money(med['median_max'])}, median "
                         f"midpoint {_fmt_money(med['median_mid'])}")
    else:
        lines.append("No comparable ranges to take a median over.")
    counts = data["counts"]
    parts = []
    for key, label in (("meets_target", "meet the target"),
                       ("below_target", "below the target"),
                       ("above_target", "above the target"),
                       ("not_compared", "not compared"),
                       ("without_range", "with no range listed")):
        if counts[key]:
            parts.append(f"{counts[key]} {label}")
    if parts:
        lines.append(", ".join(parts) + ".")
    lines.append("")
    if not rows:
        lines.append("No postings to analyse"
                     + (" above the floor" if data["min_pay"] is not None
                        else "") + ".")
    for i, row in enumerate(rows, 1):
        line = (f"  {i}. {row['title']} — {row['company']}  "
                f"{_range_text(row)}  {row['verdict']}")
        if row["verdict_reason"]:
            line += f" ({row['verdict_reason']})"
        if row["applied"]:
            line += f"  [already applied as {row['applied_id']}]"
        lines.append(line)
    if data["skipped"]:
        lines.append("")
        lines.append("Skipped files (could not be read):")
        for item in data["skipped"]:
            lines.append(f"  {item['file']}: {item['reason']}")
    return "\n".join(lines) + "\n"
