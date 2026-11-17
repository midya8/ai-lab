"""Company dossier: everything you know about one employer, on one page.

``autojob company NAME`` pulls together what the other commands
already recorded about a company — its applications and how far each
one got, the people you spoke to there, the interview rounds, the
offers, the notes, the debriefs, and why any of its applications were
rejected — plus whether the company is on the blocklist. ``autojob
company`` with no name lists every company in the log instead, most
applications first, so you can see at a glance where your search is
concentrated.

The honest rules, carried over from the other modules:

- A dossier is a record, not a to-do list. It shows applications in
  every status, including rejected and withdrawn ones, and the people
  and notes on them: a company that turned you down is still part of
  the history, and hiding it would rewrite the search. (The contacts
  and notes lists leave terminal applications out because they are
  about what to do next; this command is about what happened.)
- Nothing is inferred. The match-score average covers only the
  applications that recorded a score, and the count of unscored ones
  is printed next to it. The best stage reached comes from the
  status history, the same ``reached_stages`` the funnel uses. A
  company with no applications shows only its blocklist record, and
  a name with neither an application nor a blocklist record is an
  error, not an empty page — an empty dossier would look like a
  company you never touched.
- Names are matched the way the blocklist matches them: case and
  extra spaces ignored, so "vector labs" and "Vector Labs" are one
  company, and the dossier shows the spelling the log first used.
  People are grouped the same way the contacts list groups them:
  one row per person, latest touch, touch count, first-contact date,
  with the role, channel, and note falling back to the latest record
  that has them.
- A debrief is paired with its round by date and round label, the
  same pairing the timeline uses; a write-up for a round that was
  never recorded as an interview is listed and flagged unmatched,
  not hidden. A note with no parseable date or no text, or a contact
  with no parseable date, is skipped, never fatal — a hand-edited
  log line must not break the whole dossier.
- The command is read-only: it records nothing, changes nothing,
  and never touches the log. It reads the blocklist file only to
  show whether the company is on it.
"""

from __future__ import annotations

import datetime
import json

from .analytics import FUNNEL_STAGES, TERMINAL_STATUSES, reached_stages
from .blocklist import is_blocked
from .contacts import DEFAULT_QUIET_DAYS
from .interview import collect_interviews
from .offer import first_year_total, latest_offer

_STAGE_RANK = {stage: index for index, stage in enumerate(FUNNEL_STAGES)}


def _parse_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError:
        return None


def normalize_company(company):
    """The grouping key for a company: words only, case ignored."""
    return " ".join(str(company).split()).casefold()


def _best_stage(entries):
    """The furthest funnel stage any of these applications reached."""
    best = None
    for entry in entries:
        for stage in reached_stages(entry):
            if best is None or _STAGE_RANK[stage] > _STAGE_RANK[best]:
                best = stage
    return best


def _score_summary(entries):
    """Average / lowest / highest over the scores actually recorded."""
    scores = [entry["match_score"] for entry in entries
              if isinstance(entry.get("match_score"), (int, float))
              and not isinstance(entry.get("match_score"), bool)]
    unscored = len(entries) - len(scores)
    if not scores:
        return {"count": 0, "unscored": unscored, "average": None,
                "lowest": None, "highest": None}
    return {"count": len(scores), "unscored": unscored,
            "average": round(sum(scores) / len(scores), 3),
            "lowest": min(scores), "highest": max(scores)}


def _application_rows(entries):
    """One row per application, newest applied first.

    An entry with no parseable ``applied_at`` sorts last, never
    fatal — the log may have been hand-edited.
    """
    rows = []
    for entry in entries:
        rejection = entry.get("rejection")
        if not isinstance(rejection, dict):
            rejection = None
        else:
            rejection = {
                "category": (rejection.get("category")
                             or rejection.get("reason")),
                "note": rejection.get("note"),
                "recorded_at": rejection.get("recorded_at"),
            }
        resume = entry.get("resume")
        resume_path = None
        if isinstance(resume, dict):
            resume_path = resume.get("path")
        elif isinstance(resume, str):
            resume_path = resume
        rows.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "status": entry.get("status"),
            "applied_at": entry.get("applied_at"),
            "match_score": entry.get("match_score"),
            "source": entry.get("source"),
            "referrer": entry.get("referrer"),
            "deadline": entry.get("deadline"),
            "resume": resume_path,
            "best_stage": _best_stage([entry]),
            "rejection": rejection,
        })
    # Newest applied first; an entry with no parseable applied_at
    # sorts last, never fatal. The sort is stable, so rows on the
    # same day keep their log order.
    dated = [row for row in rows
             if _parse_date(row["applied_at"]) is not None]
    undated = [row for row in rows
               if _parse_date(row["applied_at"]) is None]
    dated.sort(key=lambda row: _parse_date(row["applied_at"]),
               reverse=True)
    return dated + undated


def _people_rows(entries, today, quiet_days):
    """Everyone spoken to at this company, quietest first.

    The same grouping as the contacts list — one row per person,
    latest touch and touch count — but across every application for
    this company, and including terminal ones: the dossier is a
    record of what happened, not a list of who to nudge.
    """
    people = {}
    for entry in entries:
        for contact in entry.get("contacts") or []:
            if not isinstance(contact, dict):
                continue
            name = contact.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            key = " ".join(name.split()).lower()
            people.setdefault(key, []).append((entry, contact))
    rows = []
    for records in people.values():
        dated = [(entry, rec, _parse_date(rec.get("date")))
                 for entry, rec in records]
        dated = [(entry, rec, when) for entry, rec, when in dated
                 if when is not None]
        if not dated:
            continue
        dated.sort(key=lambda triple: triple[2])
        first_entry, first, _first_when = dated[0]
        latest_entry, latest, last_touch = dated[-1]
        role = latest.get("role")
        channel = latest.get("channel")
        note = latest.get("note")
        if role is None or channel is None or note is None:
            for _entry, rec, _when in reversed(dated):
                if role is None:
                    role = rec.get("role")
                if channel is None:
                    channel = rec.get("channel")
                if note is None:
                    note = rec.get("note")
                if (role is not None and channel is not None
                        and note is not None):
                    break
        days_since = (today - last_touch).days
        app_ids = []
        for entry, _rec in records:
            app_id = entry.get("id")
            if app_id is not None and app_id not in app_ids:
                app_ids.append(app_id)
        rows.append({
            "name": latest.get("name") or first.get("name"),
            "role": role,
            "channel": channel,
            "note": note,
            "first_contact": first.get("date"),
            "last_contact": latest.get("date"),
            "touches": len(records),
            "days_since_touch": days_since,
            "quiet": days_since > quiet_days,
            "applications": app_ids,
        })
    rows.sort(key=lambda row: (-row["days_since_touch"],
                               (row["name"] or "").lower()))
    return rows


def _offer_rows(entries, today):
    """The latest offer on each application, soonest deadline first."""
    rows = []
    for entry in entries:
        offer = latest_offer(entry)
        if offer is None:
            continue
        deadline = offer.get("deadline")
        when = _parse_date(deadline)
        days_to_deadline = (when - today).days if when is not None else None
        try:
            first_year, equity_counted = first_year_total(offer)
        except (KeyError, TypeError):
            # A hand-edited offer without the validated shape (no
            # base_annual) must not break the dossier: the total is
            # simply not shown, the same "missing stays missing" rule
            # as the export.
            first_year, equity_counted = None, None
        rows.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "status": entry.get("status"),
            "base": offer.get("base"),
            "currency": offer.get("currency"),
            "bonus": offer.get("bonus"),
            "equity": offer.get("equity"),
            "first_year_total": first_year,
            "first_year_equity_counted": equity_counted,
            "deadline": deadline,
            "days_to_deadline": days_to_deadline,
            "work_mode": offer.get("work_mode"),
            "pto_days": offer.get("pto_days"),
            "note": offer.get("note"),
            "_deadline": when or datetime.date.max,
        })
    rows.sort(key=lambda row: (row["_deadline"],
                               (row["title"] or "").lower()))
    for row in rows:
        del row["_deadline"]
    return rows


def _note_rows(entries, today):
    """Every note on these applications, newest first.

    Notes on terminal applications are included — the dossier is a
    record — with the same cleaning as the notes list: a note with
    no parseable date or no text is skipped, never fatal. Within a
    day the latest-recorded note comes first.
    """
    rows = []
    seq = 0
    for entry in entries:
        for note in entry.get("notes") or []:
            if not isinstance(note, dict):
                continue
            when = _parse_date(note.get("date"))
            if when is None:
                continue
            text = note.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            tag = note.get("tag")
            if not isinstance(tag, str) or not tag.strip():
                tag = None
            else:
                tag = tag.strip()
            rows.append({
                "id": entry.get("id"),
                "title": entry.get("title"),
                "date": note["date"],
                "tag": tag,
                "text": text.strip(),
                "days_ago": (today - when).days,
                "_seq": seq,
            })
            seq += 1
    rows.sort(key=lambda row: (row["date"], row["_seq"]), reverse=True)
    for row in rows:
        del row["_seq"]
    return rows


def _debrief_rows(entries):
    """Every debrief on these applications, newest first.

    The interviewer is paired from the round recorded on the same
    application (same date and round label); a write-up for a round
    that was never recorded is listed and flagged unmatched.
    """
    rows = []
    for entry in entries:
        by_key = {}
        for interview in entry.get("interviews") or []:
            if not isinstance(interview, dict):
                continue
            by_key[(interview.get("date"), interview.get("round"))] = \
                interview.get("interviewer")
        for debrief in entry.get("debriefs") or []:
            if not isinstance(debrief, dict):
                continue
            when = _parse_date(debrief.get("date"))
            if when is None:
                continue
            key = (debrief.get("date"), debrief.get("round"))
            rows.append({
                "id": entry.get("id"),
                "title": entry.get("title"),
                "date": debrief["date"],
                "round": debrief.get("round"),
                "rating": debrief.get("rating"),
                "interviewer": by_key.get(key),
                "matched": key in by_key,
                "went_well": debrief.get("went_well"),
                "improve": debrief.get("improve"),
                "questions": debrief.get("questions") or [],
            })
    rows.sort(key=lambda row: row["date"], reverse=True)
    return rows


def _rejection_rows(entries):
    """Every recorded rejection reason, most recently recorded first."""
    rows = []
    for entry in entries:
        rejection = entry.get("rejection")
        if not isinstance(rejection, dict):
            continue
        rows.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "category": (rejection.get("category")
                         or rejection.get("reason")),
            "note": rejection.get("note"),
            "recorded_at": rejection.get("recorded_at"),
        })
    rows.sort(key=lambda row: (row["recorded_at"] is None,
                               row["recorded_at"] or ""),
              reverse=True)
    return rows


def build_dossier(entries, company, today=None, quiet_days=DEFAULT_QUIET_DAYS,
                  blocklist=None):
    """Everything recorded about one company, on one page.

    Returns a dict with "found" False when the company has no
    application in the log and no blocklist record — the caller
    turns that into an error rather than an empty page.
    """
    if today is None:
        today = datetime.date.today()
    if quiet_days is None:
        quiet_days = DEFAULT_QUIET_DAYS
    if isinstance(company, str):
        wanted = normalize_company(company)
    else:
        wanted = ""
    matched = [entry for entry in entries
               if isinstance(entry, dict)
               and isinstance(entry.get("company"), str)
               and normalize_company(entry["company"]) == wanted]
    blocked = is_blocked(blocklist, company)
    display = (matched[0]["company"] if matched
               else (company.strip() if isinstance(company, str)
                     and company.strip() else ""))
    if not matched and blocked is None:
        return {"company": display, "found": False, "applications": [],
                "summary": {"applications": 0, "live": 0, "terminal": 0,
                            "best_stage": None,
                            "scores": {"count": 0, "unscored": 0,
                                       "average": None, "lowest": None,
                                       "highest": None},
                            "sources": {}, "referrers": []},
                "people": [], "interviews": {"upcoming": [], "past": [],
                                             "counts": {"upcoming": 0,
                                                        "past": 0,
                                                        "today": 0}},
                "offers": [], "notes": [], "debriefs": [],
                "rejections": [], "blocked": None}
    sources = {}
    referrers = []
    for entry in matched:
        source = entry.get("source")
        if isinstance(source, str) and source.strip():
            sources[source] = sources.get(source, 0) + 1
        referrer = entry.get("referrer")
        if isinstance(referrer, str) and referrer.strip() \
                and referrer.strip() not in referrers:
            referrers.append(referrer.strip())
    return {
        "company": display,
        "found": True,
        "applications": _application_rows(matched),
        "summary": {
            "applications": len(matched),
            "live": sum(1 for entry in matched
                        if entry.get("status") not in TERMINAL_STATUSES),
            "terminal": sum(1 for entry in matched
                            if entry.get("status") in TERMINAL_STATUSES),
            "best_stage": _best_stage(matched),
            "scores": _score_summary(matched),
            "sources": dict(sorted(sources.items())),
            "referrers": referrers,
        },
        "people": _people_rows(matched, today, quiet_days),
        "interviews": collect_interviews(matched, today=today),
        "offers": _offer_rows(matched, today),
        "notes": _note_rows(matched, today),
        "debriefs": _debrief_rows(matched),
        "rejections": _rejection_rows(matched),
        "blocked": ({"company": blocked["company"],
                     "reason": blocked.get("reason"),
                     "blocked_on": blocked.get("blocked_on")}
                    if blocked is not None else None),
    }


def list_companies(entries, blocklist=None):
    """Every company in the log, most applications first.

    A blocklisted company with no applications is listed too, with
    zero applications — it is still a company you have a position
    on. Returns {"companies": [...], "count": n}.
    """
    by_key = {}
    order = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        company = entry.get("company")
        if not isinstance(company, str) or not company.strip():
            continue
        key = normalize_company(company)
        if key not in by_key:
            by_key[key] = {"company": company.strip(), "entries": []}
            order.append(key)
        by_key[key]["entries"].append(entry)
    blocked = blocklist.get("blocked", []) \
        if isinstance(blocklist, dict) else (blocklist or [])
    for record in blocked:
        if not isinstance(record, dict):
            continue
        company = record.get("company")
        if not isinstance(company, str) or not company.strip():
            continue
        key = normalize_company(company)
        if key not in by_key:
            by_key[key] = {"company": company.strip(), "entries": []}
            order.append(key)
    rows = []
    for key in order:
        group = by_key[key]
        group_entries = group["entries"]
        applied = [_parse_date(entry.get("applied_at"))
                   for entry in group_entries]
        applied = [when for when in applied if when is not None]
        statuses = {}
        for entry in group_entries:
            status = entry.get("status")
            if isinstance(status, str) and status:
                statuses[status] = statuses.get(status, 0) + 1
        rows.append({
            "company": group["company"],
            "applications": len(group_entries),
            "best_stage": _best_stage(group_entries),
            "latest_applied": max(applied).isoformat() if applied else None,
            "blocked": is_blocked(blocked, group["company"]) is not None,
            "statuses": dict(sorted(statuses.items())),
        })
    rows.sort(key=lambda row: (-row["applications"],
                               row["company"].casefold()))
    return {"companies": rows, "count": len(rows)}


def company_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def companies_json(data):
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _fmt_application(row):
    head = f"{row['id']}: {row['title']} — {row['status']}"
    if row["applied_at"]:
        head += f" (applied {row['applied_at']})"
    if row["match_score"] is not None:
        head += f", match {row['match_score']:.3f}"
    if row["best_stage"]:
        head += f", reached {row['best_stage']}"
    if row["source"]:
        head += f", via {row['source']}"
    if row["referrer"]:
        head += f" (referred by {row['referrer']})"
    if row["deadline"]:
        head += f", closes {row['deadline']}"
    if row["resume"]:
        head += f", resume {row['resume']}"
    if row["rejection"] and row["rejection"].get("category"):
        head += f" — rejected: {row['rejection']['category']}"
        if row["rejection"].get("note"):
            head += f" ({row['rejection']['note']})"
    return head


def _fmt_person(row):
    head = f"{row['name']}"
    if row["role"]:
        head += f", {row['role']}"
    if row["channel"]:
        head += f" ({row['channel']})"
    head += (f" — {row['touches']} touch"
             + ("es" if row["touches"] != 1 else ""))
    head += f", last spoke {row['last_contact']}"
    if row["days_since_touch"] == 0:
        head += " (today)"
    elif row["days_since_touch"] == 1:
        head += " (yesterday)"
    else:
        head += f" ({row['days_since_touch']}d ago)"
    if row["quiet"]:
        head += " — quiet"
    if row["note"]:
        head += f": {row['note']}"
    return head


def _fmt_interview(row):
    head = f"{row['date']}: {row['round']} — {row['title']}"
    if row["interviewer"]:
        head += f" with {row['interviewer']}"
    if row["mode"]:
        head += f" ({row['mode']})"
    if row["duration_minutes"]:
        head += f", {row['duration_minutes']} min"
    if row["time"]:
        head += f" at {row['time']}"
    if row["days_until"] is not None:
        if row["days_until"] == 0:
            head += " — today"
        elif row["days_until"] == 1:
            head += " — tomorrow"
        else:
            head += f" — in {row['days_until']}d"
    elif row["days_ago"] is not None:
        if row["days_ago"] == 0:
            head += " — today"
        elif row["days_ago"] == 1:
            head += " — yesterday"
        else:
            head += f" — {row['days_ago']}d ago"
    if row["notes"]:
        head += f": {row['notes']}"
    return head


def _fmt_money(value):
    """Money the way the offer list prints it: whole figures with no
    trailing .0, anything else to two decimals."""
    if value is None:
        return "—"
    if value == int(value):
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _fmt_offer(row):
    head = f"{row['title']}: "
    if row["base"] is not None:
        head += f"{_fmt_money(row['base'])}"
        if row["currency"]:
            head += f" {row['currency']}"
    else:
        head += "no base recorded"
    if row["bonus"]:
        head += f" + {_fmt_money(row['bonus'])} bonus"
    if row["equity"]:
        head += f" + {_fmt_money(row['equity'])} equity"
    if row["first_year_total"] is not None:
        head += f" (first year {_fmt_money(row['first_year_total'])}"
        if row["equity"] and not row["first_year_equity_counted"]:
            head += ", equity not counted — no vesting schedule"
        head += ")"
    if row["work_mode"]:
        head += f", {row['work_mode']}"
    if row["deadline"]:
        head += f", decide by {row['deadline']}"
        if row["days_to_deadline"] is not None:
            if row["days_to_deadline"] < 0:
                head += f" ({-row['days_to_deadline']}d overdue)"
            elif row["days_to_deadline"] == 0:
                head += " (today)"
            else:
                head += f" ({row['days_to_deadline']}d left)"
    if row["note"]:
        head += f": {row['note']}"
    return head


def _fmt_note(row):
    head = f"{row['date']}: {row['text']}"
    if row["tag"]:
        head += f" [{row['tag']}]"
    return head


def _fmt_debrief(row):
    head = f"{row['date']}: {row['round']} — {row['title']}"
    if row["interviewer"]:
        head += f" with {row['interviewer']}"
    if row["rating"] is not None:
        head += f", rated {row['rating']} of 5"
    if not row["matched"]:
        head += " (no matching interview recorded)"
    if row["went_well"]:
        head += f" — went well: {row['went_well']}"
    if row["improve"]:
        head += f" — improve: {row['improve']}"
    if row["questions"]:
        head += f" — asked: {'; '.join(row['questions'])}"
    return head


def render_company_markdown(data):
    """The dossier as readable text: applications, people, rounds,
    offers, notes, write-ups, rejections, and the blocklist."""
    lines = [f"Company — {data['company']}"]
    summary = data["summary"]
    if data["blocked"] is not None:
        blocked = data["blocked"]
        lines.append("Blocked — " + (blocked["reason"]
                                     or "no reason given")
                     + f" (blocked {blocked['blocked_on']})")
    counts = f"{summary['applications']} application"
    if summary["applications"] != 1:
        counts += "s"
    counts += f" ({summary['live']} live, {summary['terminal']} closed)"
    if summary["best_stage"]:
        counts += f", furthest stage reached: {summary['best_stage']}"
    lines.append(counts)
    scores = summary["scores"]
    if scores["count"]:
        line = (f"Match scores: {scores['count']} scored, average "
                f"{scores['average']:.3f}, range {scores['lowest']:.3f}–"
                f"{scores['highest']:.3f}")
        if scores["unscored"]:
            line += f", {scores['unscored']} unscored"
        lines.append(line)
    elif summary["applications"]:
        lines.append("Match scores: none recorded")
    if summary["sources"]:
        sources = ", ".join(f"{channel} ({n})" for channel, n
                            in summary["sources"].items())
        lines.append(f"Sources: {sources}")
    if summary["referrers"]:
        lines.append("Referred by: " + ", ".join(summary["referrers"]))
    lines.append("")
    lines.append("Applications")
    if not data["applications"]:
        lines.append("  No applications — only the blocklist record above.")
    for row in data["applications"]:
        lines.append(f"  {_fmt_application(row)}")
    lines.append("")
    lines.append("People")
    if not data["people"]:
        lines.append("  Nobody recorded for this company.")
    for row in data["people"]:
        lines.append(f"  {_fmt_person(row)}")
    interviews = data["interviews"]
    lines.append("")
    lines.append("Interviews")
    if not interviews["upcoming"] and not interviews["past"]:
        lines.append("  No rounds recorded for this company.")
    for row in interviews["upcoming"]:
        lines.append(f"  {_fmt_interview(row)}")
    for row in interviews["past"]:
        lines.append(f"  {_fmt_interview(row)}")
    lines.append("")
    lines.append("Offers")
    if not data["offers"]:
        lines.append("  No offers recorded for this company.")
    for row in data["offers"]:
        lines.append(f"  {_fmt_offer(row)}")
    lines.append("")
    lines.append("Notes")
    if not data["notes"]:
        lines.append("  Nothing written down about this company.")
    for row in data["notes"]:
        lines.append(f"  {_fmt_note(row)}")
    lines.append("")
    lines.append("Debriefs")
    if not data["debriefs"]:
        lines.append("  No interview write-ups for this company.")
    for row in data["debriefs"]:
        lines.append(f"  {_fmt_debrief(row)}")
    lines.append("")
    lines.append("Rejections")
    if not data["rejections"]:
        lines.append("  No rejection reasons recorded for this company.")
    for row in data["rejections"]:
        head = f"  {row['id']}: {row['title']} — {row['category'] or 'no reason recorded'}"
        if row["note"]:
            head += f" ({row['note']})"
        if row["recorded_at"]:
            head += f", recorded {row['recorded_at']}"
        lines.append(head)
    return "\n".join(lines) + "\n"


def render_companies_markdown(data):
    """The company list: every employer in the log, most
    applications first, with the blocklisted ones flagged."""
    rows = data["companies"]
    word = "company" if data["count"] == 1 else "companies"
    lines = [f"Companies — {data['count']} {word} in the log"]
    if not rows:
        lines.append("No applications logged yet — apply to a posting "
                     "with `apply`, and the company shows up here.")
        return "\n".join(lines) + "\n"
    for row in rows:
        head = f"  {row['company']} — {row['applications']} application"
        if row["applications"] != 1:
            head += "s"
        if row["best_stage"]:
            head += f", reached {row['best_stage']}"
        if row["latest_applied"]:
            head += f", latest {row['latest_applied']}"
        if row["statuses"]:
            statuses = ", ".join(f"{status} ({n})" for status, n
                                 in row["statuses"].items())
            head += f" [{statuses}]"
        if row["blocked"]:
            head += " — BLOCKED"
        lines.append(head)
    return "\n".join(lines) + "\n"
