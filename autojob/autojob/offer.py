"""Offer comparison: record job offers and compare them side by side.

``offer --record`` stores an offer on a tracker entry (base, bonus,
equity, time off, work mode, start date, decision deadline). ``offer``
on its own compares every recorded offer against the salary target in
the profile's preferences and ranks the comparable ones.

The honest rules, carried over from salary.py:

- Amounts are compared as stated. The base is annualised only when the
  offer states a period: hourly at 40 hours a week for 52 weeks (2,080
  hours), monthly at 12 months — the assumption is printed on every
  report, and the stated figure is kept alongside the annualised one.
- A bonus is assumed annual; the report says so. Equity counts towards
  the first-year total only when a vesting schedule is given
  (equity / vesting years per year); equity with no vesting schedule is
  listed, never counted.
- Currencies are never converted. A verdict and a rank happen only when
  the offer states the same three-letter currency code as the profile
  target; anything else is listed with the reason, not guessed.
- The verdict compares the annual base against the target, exactly as
  salary analysis compares a posting's range: below when the base is
  under the target minimum, above when it is over the target maximum,
  meets otherwise. Ranking is by first-year total (base + bonus +
  equity per vesting year), so the best-paying offer can still be
  below target on base alone, and the report says so.
- The deadline is a fact, not a refusal: an offer past its deadline is
  flagged "deadline passed", with days to the deadline on every row.
- One row per application, using its latest recorded offer; the number
  of offers recorded for the application is shown when it is more than
  one. Recording an offer does not move the application's status —
  status changes stay explicit through ``status``.
"""

from __future__ import annotations

import datetime
import json

from .salary import CURRENCY_CODES, _comparable, target_from_profile

WORK_MODES = ("remote", "hybrid", "onsite")
PERIODS = ("annual", "hourly", "monthly")
HOURS_PER_YEAR = 2080  # 40 hours a week, 52 weeks


class OfferError(ValueError):
    """An offer that cannot be recorded as stated."""


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_date(value, field, errors):
    """A YYYY-MM-DD date, or None when the value is empty."""
    if value is None or value == "":
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        errors.append(f"{field} must be YYYY-MM-DD, got {value!r}")
        return None


def validate_offer(data, today=None):
    """Validate an offer dict and return the cleaned, stored form.

    Every problem is reported at once, like profile validation. The
    returned dict keeps the figures as stated (``base``, ``period``)
    plus the annualised base (``base_annual``) the comparison uses.
    ``today`` defaults to the real current date; pass an explicit date
    in tests.
    """
    if today is None:
        today = datetime.date.today().isoformat()
    errors = []
    data = data or {}

    base = data.get("base")
    if base is None:
        errors.append("base is required")
    elif not _is_number(base) or base <= 0:
        errors.append(f"base must be a number above 0, got {base!r}")

    currency = data.get("currency")
    if currency is None:
        errors.append("currency is required")
    elif (not isinstance(currency, str)
            or len(currency) != 3 or not currency.isalpha()
            or currency.upper() != currency):
        errors.append(f"currency must be a 3-letter code such as CAD, "
                      f"got {currency!r}")
    else:
        currency = currency.upper()

    period = data.get("period") or "annual"
    if period not in PERIODS:
        errors.append(f"period must be one of {', '.join(PERIODS)}, "
                      f"got {period!r}")

    bonus = data.get("bonus")
    if bonus is not None and (not _is_number(bonus) or bonus < 0):
        errors.append(f"bonus must be 0 or more, got {bonus!r}")

    equity = data.get("equity")
    if equity is not None and (not _is_number(equity) or equity < 0):
        errors.append(f"equity must be 0 or more, got {equity!r}")

    vesting_years = data.get("vesting_years")
    if vesting_years is not None:
        if (not isinstance(vesting_years, int)
                or isinstance(vesting_years, bool)
                or vesting_years <= 0):
            errors.append(f"vesting_years must be a whole number above 0, "
                          f"got {vesting_years!r}")

    pto_days = data.get("pto_days")
    if pto_days is not None:
        if (not isinstance(pto_days, int) or isinstance(pto_days, bool)
                or pto_days < 0):
            errors.append(f"pto_days must be a whole number 0 or more, "
                          f"got {pto_days!r}")

    work_mode = data.get("work_mode")
    if work_mode is not None and work_mode not in WORK_MODES:
        errors.append(f"work_mode must be one of {', '.join(WORK_MODES)}, "
                      f"got {work_mode!r}")

    start_date = _parse_date(data.get("start_date"), "start_date", errors)
    deadline = _parse_date(data.get("deadline"), "deadline", errors)

    note = data.get("note")
    if note is not None and not isinstance(note, str):
        errors.append(f"note must be text, got {note!r}")

    if errors:
        raise OfferError("; ".join(errors))

    if period == "hourly":
        base_annual = base * HOURS_PER_YEAR
    elif period == "monthly":
        base_annual = base * 12
    else:
        base_annual = base

    offer = {
        "base": base,
        "currency": currency,
        "period": period,
        "base_annual": base_annual,
        "bonus": bonus,
        "equity": equity,
        "vesting_years": vesting_years,
        "pto_days": pto_days,
        "work_mode": work_mode,
        "start_date": start_date.isoformat() if start_date else None,
        "deadline": deadline.isoformat() if deadline else None,
        "note": note,
        "recorded_at": today,
    }
    return offer


def total_cash(offer):
    """Annual base plus annual bonus (a missing bonus counts as 0)."""
    return offer["base_annual"] + (offer.get("bonus") or 0)


def first_year_total(offer):
    """(total, equity_counted) for the first year.

    Equity counts only when a vesting schedule is given, as
    equity / vesting years per year; equity with no vesting schedule
    is listed, never counted.
    """
    total = total_cash(offer)
    equity = offer.get("equity")
    vesting = offer.get("vesting_years")
    if equity is not None and vesting:
        return total + equity / vesting, True
    return total, False


def latest_offer(entry):
    """The entry's most recent offer, or None when it has none.

    Ties on recorded date keep the last recorded offer in the list.
    """
    offers = entry.get("offers") or []
    if not offers:
        return None
    return max(enumerate(offers),
               key=lambda pair: (pair[1].get("recorded_at") or "", pair[0]))[1]


def _verdict(offer, target):
    """(verdict, reason) for one offer's annual base against the target.

    Verdicts: "meets target", "below target", "above target",
    "no target set", "not compared". A verdict of "not compared"
    always carries the reason why.
    """
    if target is None:
        return "no target set", None
    currency = offer.get("currency")
    target_currency = target.get("currency")
    if not _comparable(currency, target_currency):
        if currency not in CURRENCY_CODES:
            reason = "offer does not state a known currency code"
        elif target_currency not in CURRENCY_CODES:
            reason = "profile target has no currency"
        else:
            reason = (f"offer is {currency}, "
                      f"target is {target_currency}")
        return "not compared", reason
    if target.get("min") is not None and offer["base_annual"] < target["min"]:
        return "below target", None
    if target.get("max") is not None and offer["base_annual"] > target["max"]:
        return "above target", None
    return "meets target", None


def compare_offers(entries, profile=None, today=None):
    """Compare every recorded offer against the profile's salary target.

    Returns a dict with the target, one row per application that has an
    offer (its latest), verdict counts, and the best comparable offer.
    Rows are ranked by first-year total, but only among offers in the
    target's currency — anything else carries ``rank: None`` and the
    reason it could not be ranked. With no target at all, nothing is
    ranked: the rows are listed, not ordered by pay.
    """
    if today is None:
        today = datetime.date.today()
    target = target_from_profile(profile) if profile else None
    rows = []
    n_offers = 0
    for entry in entries:
        offers = entry.get("offers") or []
        n_offers += len(offers)
        if not offers:
            continue
        offer = latest_offer(entry)
        cash = total_cash(offer)
        first_year, equity_counted = first_year_total(offer)
        verdict, reason = _verdict(offer, target)
        comparable = verdict != "not compared" and target is not None
        deadline = offer.get("deadline")
        days_to_deadline = None
        if deadline:
            days_to_deadline = (datetime.date.fromisoformat(deadline)
                                - today).days
        rows.append({
            "id": entry["id"],
            "company": entry["company"],
            "title": entry["title"],
            "status": entry["status"],
            "offer": offer,
            "total_cash": cash,
            "first_year_total": first_year,
            "equity_counted": equity_counted,
            "verdict": verdict,
            "verdict_reason": reason,
            "comparable": comparable,
            "rank": None,
            "days_to_deadline": days_to_deadline,
            "deadline_passed": (days_to_deadline is not None
                                and days_to_deadline < 0),
            "n_offers": len(offers),
        })

    ranked = [row for row in rows if row["comparable"]]
    ranked.sort(key=lambda row: (-row["first_year_total"],
                                 -row["total_cash"],
                                 row["company"].lower(),
                                 row["title"].lower()))
    for i, row in enumerate(ranked, 1):
        row["rank"] = i

    def order(row):
        return (0 if row["rank"] is not None else 1,
                row["rank"] if row["rank"] is not None else 0,
                row["company"].lower(), row["title"].lower())

    rows.sort(key=order)
    best = ranked[0] if ranked else None
    counts = {
        "applications_with_offers": len(rows),
        "offers": n_offers,
        "comparable": len(ranked),
        "meets_target": sum(1 for r in rows if r["verdict"] == "meets target"),
        "below_target": sum(1 for r in rows if r["verdict"] == "below target"),
        "above_target": sum(1 for r in rows if r["verdict"] == "above target"),
        "not_compared": sum(1 for r in rows if r["verdict"] == "not compared"),
        "no_target": sum(1 for r in rows if r["verdict"] == "no target set"),
        "deadline_passed": sum(1 for r in rows if r["deadline_passed"]),
    }
    return {
        "target": target,
        "rows": rows,
        "counts": counts,
        "best": best,
    }


def offer_json(data):
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


def render_offer_markdown(data):
    """Human-readable offer comparison, best comparable offer first."""
    rows = data["rows"]
    counts = data["counts"]
    lines = [f"Offer comparison — {counts['applications_with_offers']} "
             f"application(s) with offers ({counts['offers']} offers)"]
    target_text = _target_text(data["target"])
    if target_text is None:
        lines.append("No salary target in the profile — offers are listed, "
                     "not ranked or compared.")
    else:
        lines.append(f"Target: {target_text} (verdicts compare the annual "
                     f"base; ranking is by first-year total)")
    lines.append("A bonus is assumed annual. Equity counts towards the "
                 "first-year total only when a vesting schedule is given "
                 "(equity / vesting years per year); equity with no vesting "
                 "schedule is listed, not counted. An hourly base is "
                 "annualised at 2,080 hours a year, a monthly base at 12 "
                 "months. Currencies are never converted — an offer is "
                 "ranked only when it states the same currency code as "
                 "the target.")
    lines.append("")
    best = data["best"]
    if best is not None:
        offer = best["offer"]
        lines.append(f"Best comparable offer: {best['title']} — "
                     f"{best['company']}  {offer['currency']} "
                     f"{_fmt_money(best['first_year_total'])} first year "
                     f"(base {_fmt_money(offer['base_annual'])} + bonus "
                     f"{_fmt_money(offer.get('bonus') or 0)}"
                     + (f" + equity {_fmt_money(offer['equity'] / offer['vesting_years'])}"
                        if best["equity_counted"] else "")
                     + f")  {best['verdict']}")
    elif rows:
        lines.append("No offer is comparable with the target, so none is ranked.")
    else:
        lines.append("No offers recorded yet — use "
                     "`offer --record --id APP-001 --base 130000 "
                     "--currency CAD` to record one.")
    if not rows:
        return "\n".join(lines) + "\n"
    lines.append("")
    for row in rows:
        offer = row["offer"]
        currency = offer["currency"]
        line = (f"  {row['rank'] if row['rank'] is not None else '—'}. "
                f"{row['title']} — {row['company']}  "
                f"{currency} {_fmt_money(row['first_year_total'])} first year "
                f"(base {_fmt_money(offer['base_annual'])}"
                + (f" as stated {currency} "
                   f"{_fmt_money(offer['base'])}/{offer['period']}"
                   if offer["period"] != "annual" else "")
                + f", bonus {_fmt_money(offer.get('bonus') or 0)}"
                + (f", equity {currency} {_fmt_money(offer['equity'])}"
                   + (f" over {offer['vesting_years']}y"
                      if offer.get("vesting_years") else "")
                   if offer.get("equity") is not None else "")
                + ("" if row["equity_counted"] else
                   " (equity not counted: no vesting schedule)"
                   if offer.get("equity") is not None else "")
                + f")  {row['verdict']}")
        if row["verdict_reason"]:
            line += f" ({row['verdict_reason']})"
        if offer.get("pto_days") is not None:
            line += f"  PTO {offer['pto_days']}d"
        if offer.get("work_mode"):
            line += f"  {offer['work_mode']}"
        if row["days_to_deadline"] is not None:
            if row["deadline_passed"]:
                line += f"  DEADLINE PASSED ({-row['days_to_deadline']}d ago)"
            elif row["days_to_deadline"] == 0:
                line += "  deadline today"
            else:
                line += f"  decides in {row['days_to_deadline']}d"
        if row["n_offers"] > 1:
            line += f"  [{row['n_offers']} offers recorded, latest shown]"
        lines.append(line)
    return "\n".join(lines) + "\n"
