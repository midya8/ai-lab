"""Offer negotiation brief: what you hold, what it is worth, what to say.

``autojob negotiate --id APP-001`` builds the brief for one recorded
offer: the offer as recorded, how it stands against the salary target
in the profile and against every other offer in the log, the posting's
stated range when the posting is supplied, the strengths to cite (the
matched skills with years and the verbatim bullets that prove them),
the gaps to never claim, the other live applications still in play,
and the days left to decide.

The brief states facts; it does not suggest a number. There is no
recommended ask anywhere in it — the numbers to know are the offer's
first-year total, the target, the best comparable offer in the log,
and the posting's stated range, and the ask is yours to choose from
those. A brief that invented a target number would be the same
invention the rest of autojob refuses to make.

The honest rules, carried over from ``offer`` and ``tailor``:

- The brief is built from the log and the profile as they are now.
  The leverage section reuses the tailor's skills matrix, built from
  the skills the application recorded at apply time against the
  profile as it stands today — if the profile changed since the
  application, a skill can move between the two sides, and the brief
  says which profile it used.
- Currencies are never converted. A verdict and a rank happen only
  when the offer states the same three-letter currency code as the
  profile target; anything else is listed with the reason, not
  guessed. The same rule covers the posting's stated range.
- A bonus is assumed annual and equity counts only with a vesting
  schedule, the same assumptions ``offer`` makes, and the brief says
  so. An hourly base is annualised at 2,080 hours a year, a monthly
  base at 12 months, and the stated figure is kept next to the
  annualised one.
- An offer on a terminal application (hired / rejected / withdrawn)
  is no longer live, so there is nothing to negotiate — the command
  refuses it, the same rule as every other recording command. An
  application with no offer recorded is refused too, with the command
  that fixes it.
- Walk-away options are the other applications at interview or offer
  stage — the ones still in play. Screening-stage applications are
  not listed: an application that has not reached an interview is not
  leverage yet.
- The command is read-only: it records nothing and never touches the
  log.
"""

from __future__ import annotations

import datetime
import json

from .analytics import TERMINAL_STATUSES
from .offer import compare_offers, first_year_total, latest_offer, total_cash
from .salary import analyze_salaries, target_from_profile
from .tailor import build_tailor
from .tracker import _norm

WALK_AWAY_STATUSES = ("interview", "offer")
_LEAD_SKILL_LIMIT = 5


class NegotiationError(ValueError):
    """A negotiation brief that cannot be built as stated."""


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def _match_verdict(score, required_coverage):
    """The match verdict for a recorded score and required coverage.

    Same thresholds as ``match.match_profile``: a recorded score with
    no coverage to judge is weak, not strong by default.
    """
    if score is None:
        return "unknown"
    coverage = required_coverage or 0
    if coverage >= 0.8:
        return "strong"
    if coverage >= 0.5:
        return "partial"
    return "weak"


def _leverage(entry, profile):
    """(leverage, note) for one application from the recorded skills.

    The leverage section is the tailor's skills matrix, built from
    the skills the application recorded at apply time. Returns
    (None, note) when the application recorded no skills at all —
    entries logged before v0.6 carry no skill lists — instead of
    guessing them from the posting.
    """
    keys = ("required_matched", "required_missing",
            "preferred_matched", "preferred_missing")
    if not any(key in entry for key in keys):
        return None, ("this application recorded no skills at apply "
                      "time (entries logged before v0.6 carry no skill "
                      "lists) — the strengths to cite need the "
                      "posting's skill lists")
    required_matched = list(entry.get("required_matched") or [])
    required_missing = list(entry.get("required_missing") or [])
    preferred_matched = list(entry.get("preferred_matched") or [])
    preferred_missing = list(entry.get("preferred_missing") or [])
    required_total = len(required_matched) + len(required_missing)
    required_coverage = (len(required_matched) / required_total
                         if required_total else None)
    score = entry.get("match_score")
    match = {
        "score": score,
        "verdict": _match_verdict(score, required_coverage),
        "required_coverage": required_coverage,
        "required_matched": required_matched,
        "required_missing": required_missing,
        "preferred_matched": preferred_matched,
        "preferred_missing": preferred_missing,
    }
    posting = {
        "title": entry.get("title"),
        "company": entry.get("company"),
        "location": entry.get("location"),
        "remote": entry.get("remote"),
    }
    tailor = build_tailor(profile, posting, match)
    by_skill = {row["skill"]: row for row in tailor["skills_matrix"]}
    lead_with = []
    for skill in tailor["skills_to_lead_with"][:_LEAD_SKILL_LIMIT]:
        row = by_skill.get(skill)
        if row is None:
            continue
        lead_with.append({
            "skill": row["skill"],
            "kind": row["kind"],
            "years": row.get("years"),
            "level": row.get("level"),
            "bullets": row.get("bullets") or [],
        })
    return {
        "match_score": score,
        "match_verdict": match["verdict"],
        "required_coverage": required_coverage,
        "skills_to_lead_with": lead_with,
        "n_lead_skills": len(tailor["skills_to_lead_with"]),
        "gaps": list(tailor["not_to_claim"]),
    }, None


def _walk_away(entries, entry_id):
    """Other live applications at interview or offer stage."""
    options = []
    for other in entries:
        if not isinstance(other, dict):
            continue
        if other.get("id") == entry_id:
            continue
        if other.get("status") not in WALK_AWAY_STATUSES:
            continue
        offer = latest_offer(other)
        total = None
        currency = None
        if offer is not None:
            total, _counted = first_year_total(offer)
            currency = offer.get("currency")
        options.append({
            "id": other.get("id"),
            "company": other.get("company"),
            "title": other.get("title"),
            "status": other.get("status"),
            "has_offer": offer is not None,
            "first_year_total": total,
            "currency": currency,
        })
    options.sort(key=lambda row: (
        0 if row["status"] == "offer" else 1,
        (row["company"] or "").lower(),
        (row["title"] or "").lower()))
    return options


def _posting_range(entry, entries, profile, postings_dir):
    """The posting's stated range for this application, or None.

    Returns None when no postings folder was given, a dict with
    ``found: False`` when the folder holds no posting for this
    application, and the range + verdict against the target when it
    does. A bad folder raises ValueError, the same as ``salary``.
    """
    if postings_dir is None:
        return None
    if profile is None:
        return {"found": False,
                "note": "no profile supplied — the posting's range "
                        "cannot be checked against a target"}
    data = analyze_salaries(profile, postings_dir, entries=entries)
    wanted = (_norm(entry.get("company")), _norm(entry.get("title")))
    for row in data["rows"]:
        if (_norm(row.get("company")), _norm(row.get("title"))) == wanted:
            return {
                "found": True,
                "file": row["file"],
                "salary": row["salary"],
                "period": row["period"],
                "verdict": row["verdict"],
                "verdict_reason": row["verdict_reason"],
            }
    return {"found": False,
            "note": "no posting for this application in the folder"}


def build_negotiation(entry, entries, profile=None, today=None,
                      postings_dir=None):
    """Build the negotiation brief for one application's latest offer.

    ``today`` is a ``datetime.date`` (defaults to the real current
    date; pass an explicit date in tests). ``entries`` is the whole
    log — the comparison and the walk-away options come from it.
    Raises NegotiationError when the application is terminal or has
    no offer recorded; a ValueError from a bad postings folder is
    left to the caller, the same as ``review``.
    """
    if today is None:
        today = datetime.date.today()
    if not isinstance(entry, dict):
        raise NegotiationError("application must be a log entry")
    app_id = entry.get("id")
    status = entry.get("status")
    if status in TERMINAL_STATUSES:
        raise NegotiationError(
            f"application {app_id} is {status} — its offer is no "
            "longer live, so there is nothing to negotiate")
    offer = latest_offer(entry)
    if offer is None:
        raise NegotiationError(
            f"no offer recorded for {app_id} — use "
            "`offer --record --id APP-001 --base 130000 --currency CAD` "
            "to record one first")
    clean_entries = [e for e in entries if isinstance(e, dict)]
    comparison = compare_offers(clean_entries, profile=profile, today=today)
    row = next((r for r in comparison["rows"] if r["id"] == app_id), None)
    if row is None:  # pragma: no cover - latest_offer found one above
        raise NegotiationError(f"no offer recorded for {app_id}")
    best = comparison["best"]
    target = target_from_profile(profile) if profile else None
    cash = total_cash(offer)
    first_year, equity_counted = first_year_total(offer)
    deadline = _parse_date(offer.get("deadline"))
    days_to_deadline = ((deadline - today).days
                        if deadline is not None else None)
    gap_to_target_min = None
    excess_over_target_max = None
    if row["verdict"] == "below target" and target is not None \
            and target.get("min") is not None:
        gap_to_target_min = target["min"] - offer["base_annual"]
    if row["verdict"] == "above target" and target is not None \
            and target.get("max") is not None:
        excess_over_target_max = offer["base_annual"] - target["max"]
    gap_to_best = None
    if row["comparable"] and best is not None:
        gap_to_best = best["first_year_total"] - first_year
    leverage, leverage_note = (None, None)
    if profile is not None:
        leverage, leverage_note = _leverage(entry, profile)
    else:
        leverage_note = ("no profile supplied — the strengths to cite "
                         "need the profile")
    return {
        "id": app_id,
        "company": entry.get("company"),
        "title": entry.get("title"),
        "status": status,
        "offer": offer,
        "total_cash": cash,
        "first_year_total": first_year,
        "equity_counted": equity_counted,
        "verdict": row["verdict"],
        "verdict_reason": row["verdict_reason"],
        "comparable": row["comparable"],
        "rank": row["rank"],
        "n_comparable": comparison["counts"]["comparable"],
        "best": ({
            "id": best["id"],
            "company": best["company"],
            "title": best["title"],
            "first_year_total": best["first_year_total"],
            "currency": best["offer"].get("currency"),
        } if best is not None else None),
        "gap_to_target_min": gap_to_target_min,
        "excess_over_target_max": excess_over_target_max,
        "gap_to_best_first_year": gap_to_best,
        "days_to_deadline": days_to_deadline,
        "deadline_passed": (days_to_deadline is not None
                            and days_to_deadline < 0),
        "target": target,
        "leverage": leverage,
        "leverage_note": leverage_note,
        "walk_away": _walk_away(clean_entries, app_id),
        "posting": _posting_range(entry, clean_entries, profile,
                                  postings_dir),
    }


def negotiate_json(data):
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


def render_negotiation_markdown(data):
    """Human-readable negotiation brief: the offer, the numbers, the
    strengths, the gaps, and what else is still in play."""
    offer = data["offer"]
    currency = offer.get("currency")
    lines = [f"Negotiation brief — {data['title']} at {data['company']} "
             f"({data['id']})"]
    offer_line = (f"The offer: {currency} "
                  f"{_fmt_money(offer['base_annual'])} base annual"
                  + (f" (as stated {currency} {_fmt_money(offer['base'])}"
                     f"/{offer['period']})"
                     if offer.get("period") != "annual" else "")
                  + f" + {_fmt_money(offer.get('bonus') or 0)} bonus")
    if offer.get("equity") is not None:
        offer_line += f" + {_fmt_money(offer['equity'])} equity"
        if offer.get("vesting_years"):
            offer_line += f" over {offer['vesting_years']} years"
    offer_line += (f" — first year {_fmt_money(data['first_year_total'])}"
                   + (" (equity counted)" if data["equity_counted"]
                      else " (equity not counted: no vesting schedule)"
                      if offer.get("equity") is not None else ""))
    if offer.get("pto_days") is not None:
        offer_line += f", {offer['pto_days']} days off"
    if offer.get("work_mode"):
        offer_line += f", {offer['work_mode']}"
    lines.append(offer_line)
    target_text = _target_text(data["target"])
    if target_text is None:
        lines.append("No salary target in the profile — the offer is "
                     "listed, not compared.")
    else:
        verdict_line = f"Target: {target_text} — {data['verdict']}"
        if data["verdict_reason"]:
            verdict_line += f" ({data['verdict_reason']})"
        if data["gap_to_target_min"] is not None:
            verdict_line += (f"; the base is "
                             f"{_fmt_money(data['gap_to_target_min'])} "
                             "under the target minimum")
        if data["excess_over_target_max"] is not None:
            verdict_line += (f"; the base is "
                             f"{_fmt_money(data['excess_over_target_max'])} "
                             "over the target maximum")
        lines.append(verdict_line)
    if data["comparable"]:
        if data["rank"] == 1:
            lines.append(f"Rank: 1 of {data['n_comparable']} comparable "
                         "offers — this is the best comparable offer.")
        else:
            best = data["best"]
            lines.append(f"Rank: {data['rank']} of "
                         f"{data['n_comparable']} comparable offers; the "
                         f"best is {best['title']} at {best['company']} "
                         f"({best['currency']} "
                         f"{_fmt_money(best['first_year_total'])} first "
                         "year)"
                         + (f", {_fmt_money(data['gap_to_best_first_year'])} "
                            "above this one"
                            if data["gap_to_best_first_year"] else ""))
    elif data["target"] is not None:
        lines.append("This offer is not ranked: "
                     + (data["verdict_reason"] or
                        "it cannot be compared with the target"))
    posting = data["posting"]
    if posting is None:
        lines.append("Posting range: no postings folder given — pass "
                     "`--postings-dir` to check the posting's stated range.")
    elif not posting["found"]:
        lines.append("Posting range: "
                     + (posting.get("note") or
                        "no posting for this application in the folder"))
    else:
        salary = posting["salary"]
        if salary is None:
            lines.append(f"Posting range: none listed ({posting['file']}).")
        else:
            lines.append(f"Posting range: {salary['currency']} "
                         f"{_fmt_money(salary['min'])}–"
                         f"{_fmt_money(salary['max'])} "
                         f"({posting['file']}) — {posting['verdict']}"
                         + (f" ({posting['verdict_reason']})"
                            if posting.get("verdict_reason") else ""))
    if data["days_to_deadline"] is None:
        lines.append("Decision deadline: none recorded.")
    elif data["deadline_passed"]:
        lines.append(f"Decision deadline: PASSED "
                     f"({-data['days_to_deadline']}d ago).")
    elif data["days_to_deadline"] == 0:
        lines.append("Decision deadline: today.")
    else:
        lines.append(f"Decision deadline: in "
                     f"{data['days_to_deadline']}d.")
    lines.append("A bonus is assumed annual. Equity counts towards the "
                 "first-year total only when a vesting schedule is given. "
                 "Currencies are never converted. This brief states facts "
                 "from your log and profile — it does not suggest a number "
                 "or recommend whether to accept.")
    lines.append("")
    lines.append("What you can point to")
    lines.append("")
    leverage = data["leverage"]
    if leverage is None:
        lines.append(data["leverage_note"] or
                     "No strengths listed for this application.")
    elif not leverage["skills_to_lead_with"]:
        lines.append("No matched skills to cite — the posting lists no "
                     "skills the profile has, or none were recorded.")
    else:
        shown = leverage["skills_to_lead_with"]
        for row in shown:
            label = f"- {row['skill']}"
            if row.get("years") is not None:
                label += f" ({row['years']} yrs"
                label += f", {row['level']}" if row.get("level") else ""
                label += ")"
            label += f" [{row['kind']}]"
            lines.append(label)
            for bullet in row["bullets"]:
                lines.append(f"  \"{bullet['bullet']}\" "
                             f"({bullet['title']}, {bullet['company']})")
            if not row["bullets"]:
                lines.append("  No bullet in the profile mentions this "
                            "skill — list it, but do not invent evidence "
                            "for it.")
        if leverage["n_lead_skills"] > len(shown):
            lines.append(f"  ... and "
                         f"{leverage['n_lead_skills'] - len(shown)} more "
                         "matched skill(s); the strongest five are shown.")
    if leverage is not None and leverage["gaps"]:
        lines.append("")
        lines.append("Do not claim")
        lines.append("")
        for skill in leverage["gaps"]:
            lines.append(f"- {skill}")
    lines.append("")
    lines.append("Other live options")
    lines.append("")
    if not data["walk_away"]:
        lines.append("No other live applications at interview or offer "
                     "stage.")
    else:
        for option in data["walk_away"]:
            line = (f"- {option['title']} at {option['company']} "
                    f"({option['id']}) — {option['status']} stage")
            if option["has_offer"]:
                line += (f", offer first year "
                         f"{option['currency']} "
                         f"{_fmt_money(option['first_year_total'])}")
            lines.append(line)
    lines.append("")
    return "\n".join(lines) + "\n"
