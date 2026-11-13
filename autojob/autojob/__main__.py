"""autojob CLI.

Subcommands (all offline, stdlib only):

  validate  --profile P              check a profile file
  parse     --posting J [--profile P] show the structured posting
  match     --profile P --posting J [--fail-under X] [--json]
  draft     --profile P --posting J [--min-coverage X] [--out F] [--dry-run]
  apply     --profile P --posting J [--min-coverage X] [--out-dir D] [--log L] [--dry-run]
  list      [--log L] [--status S]
  status    --log L --id APP-001 --to interview
  report    [--log L] [--json] [--stale-days N] [--today YYYY-MM-DD]
  prep      --profile P --posting J [--out F] [--json]
  tailor    --profile P --posting J [--out F] [--json]
  followup  [--log L] [--profile P] [--wait-days N] [--json] [--today YYYY-MM-DD]
  followup  --record --id APP-001 [--log L] [--note TEXT] [--today YYYY-MM-DD]
  digest    [--log L] [--json] [--out F] [--stale-days N] [--wait-days N] [--today YYYY-MM-DD]
  plan      --profile P [--log L] [--json] [--out F] [--today YYYY-MM-DD]
  rank      --profile P --postings-dir D [--log L] [--min-score X] [--include-applied] [--json] [--out F]
  salary    --profile P --postings-dir D [--log L] [--min-pay X] [--json] [--out F]
  offer     [--log L] [--profile P] [--json] [--out F] [--today YYYY-MM-DD]
  offer     --record --id APP-001 --base N --currency CAD [--bonus N]
            [--equity N --vesting-years N] [--pto-days N]
            [--mode remote|hybrid|onsite] [--period annual|hourly|monthly]
            [--start YYYY-MM-DD] [--deadline YYYY-MM-DD] [--note TEXT]
            [--today YYYY-MM-DD]
  interview [--log L] [--id APP-001] [--json] [--out F] [--today YYYY-MM-DD]
  interview --record --id APP-001 --date YYYY-MM-DD --round LABEL
            [--interviewer NAME] [--mode phone|video|onsite]
            [--duration-minutes N] [--time HH:MM] [--notes TEXT] [--today YYYY-MM-DD]
  calendar  [--log L] [--days N] [--wait-days N] [--json] [--out F] [--today YYYY-MM-DD]
  contacts  [--log L] [--id APP-001] [--json] [--out F] [--quiet-days N] [--today YYYY-MM-DD]
  contacts  --record --id APP-001 --name NAME --date YYYY-MM-DD
            [--role ROLE] [--channel email|linkedin|phone|in-person|other]
            [--note TEXT] [--today YYYY-MM-DD]
  timeline  [--log L] [--id APP-001] [--since YYYY-MM-DD]
            [--until YYYY-MM-DD] [--kind KIND ...] [--oldest] [--json]
            [--out F] [--today YYYY-MM-DD]
  deadlines [--log L] [--days N] [--overdue] [--json] [--out F] [--today YYYY-MM-DD]
  deadlines --record --id APP-001 --date YYYY-MM-DD [--today YYYY-MM-DD]
  goals     [--log L] [--goals G] [--set-weekly N] [--weeks N] [--json] [--out F]
            [--today YYYY-MM-DD]
  rejections [--log L] [--reason REASON] [--json] [--out F] [--today YYYY-MM-DD]
  rejections --record --id APP-001 --reason ghosted [--note TEXT]
            [--today YYYY-MM-DD]
  sources   [--log L] [--source CHANNEL] [--json] [--out F]
  sources   --record --id APP-001 --source referral --referrer NAME
  next      [--log L] [--days N] [--kind KIND ...] [--wait-days N]
            [--quiet-days N] [--stale-days N] [--json] [--out F]
            [--today YYYY-MM-DD]

Exit codes: 0 ok · 1 usage error · 2 profile/posting load error ·
3 match below --fail-under · 4 draft/apply refused below --min-coverage ·
5 duplicate application · 6 illegal status transition, follow-up refused,
offer refused, interview refused, contact refused, deadline refused,
rejection refused, or source refused.

autojob prepares application packets for a human to review and submit. It
never submits anything and never invents experience: every fact in a draft
comes from the profile file, and gaps are reported, not filled in.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from . import __version__
from .analytics import analytics_json, analytics_report, compute_analytics
from .calendar import build_calendar, calendar_json, render_ics
from .contacts import (ContactError, collect_contacts, contacts_for,
                       contacts_json, render_contacts_markdown,
                       validate_contact)
from .deadlines import (DeadlineError, collect_deadlines, deadlines_json,
                        render_deadlines_markdown, validate_deadline)
from .digest import build_digest, digest_json, render_digest_markdown
from .followup import due_followups, followup_json, followup_report
from .goals import (GoalsError, collect_goals, default_goals_path,
                    goals_json, load_goals, render_goals_markdown,
                    set_weekly_target)
from .interview import (InterviewError, collect_interviews,
                        interview_json, interviews_for,
                        render_interview_markdown, validate_interview)
from .letter import draft_letter, review_notes
from .match import match_profile
from .next import (NEXT_KINDS, NextError, collect_next, next_json,
                   render_next_markdown)
from .offer import (OfferError, compare_offers, offer_json,
                    render_offer_markdown, validate_offer)
from .plan import build_plan, plan_json, render_plan_markdown
from .posting import parse_posting
from .prep import build_prep, prep_json, render_prep_markdown
from .profile import ProfileError, load_profile, skill_map
from .rank import rank_json, rank_postings, render_rank_markdown
from .rejections import (REJECTION_REASONS, RejectionError,
                         collect_rejections, rejections_json,
                         render_rejections_markdown, validate_rejection)
from .report import match_json, match_report
from .salary import (analyze_salaries, render_salary_markdown, salary_json,
                     target_from_profile)
from .sources import (SOURCE_CHOICES, SourceError, collect_sources,
                      render_sources_markdown, sources_json,
                      validate_source)
from .tailor import build_tailor, render_tailor_markdown, tailor_json
from .timeline import (KINDS, collect_timeline, render_timeline_markdown,
                       timeline_json)
from .tracker import (TrackerError, add_application, find_application,
                      read_log, record_contact, record_deadline,
                      record_followup, record_interview, record_offer,
                      record_rejection, record_source, set_status, slug)

DEFAULT_OUT_DIR = "applications"


def _load_profile(path):
    try:
        return load_profile(path)
    except (ProfileError, OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot use profile {path}: {exc}", file=sys.stderr)
        return None


def _load_posting(path, profile=None):
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"error: cannot read posting {path}: {exc}", file=sys.stderr)
        return None
    profile_skills = [s["name"] for s in (profile or {}).get("skills", [])]
    return parse_posting(text, profile_skills)


def _cmd_validate(args):
    profile = _load_profile(args.profile)
    if profile is None:
        return 2
    print(f"profile OK: {profile['name']} — {profile['headline']}, "
          f"{len(profile['skills'])} skills, {len(profile['experience'])} roles")
    return 0


def _cmd_parse(args):
    profile = _load_profile(args.profile) if args.profile else None
    if args.profile and profile is None:
        return 2
    posting = _load_posting(args.posting, profile)
    if posting is None:
        return 2
    print(json.dumps(posting, indent=2, sort_keys=True))
    return 0


def _match_both(args):
    profile = _load_profile(args.profile)
    if profile is None:
        return None, None, 2
    posting = _load_posting(args.posting, profile)
    if posting is None:
        return None, None, 2
    return profile, posting, 0


def _cmd_match(args):
    profile, posting, code = _match_both(args)
    if code:
        return code
    match = match_profile(profile, posting)
    if args.json:
        print(match_json(posting, match), end="")
    else:
        print(match_report(posting, match), end="")
    if args.fail_under is not None and (match["score"] or 0) < args.fail_under:
        print(f"match score below --fail-under {args.fail_under}", file=sys.stderr)
        return 3
    return 0


def _cmd_draft(args):
    profile, posting, code = _match_both(args)
    if code:
        return code
    match = match_profile(profile, posting)
    if (match["score"] or 0) < args.min_coverage:
        print(f"refusing to draft: required coverage "
              f"{match['required_coverage']} below --min-coverage {args.min_coverage}; "
              f"missing: {', '.join(match['required_missing']) or 'none'}",
              file=sys.stderr)
        return 4
    letter = draft_letter(profile, posting, match)
    if args.dry_run or not args.out:
        print(letter, end="")
        if args.dry_run and args.out:
            print(f"(dry run: would write {args.out})", file=sys.stderr)
    else:
        Path(args.out).write_text(letter, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def _build_packet(profile, posting, match, out_dir):
    packet = Path(out_dir) / f"{slug(posting.get('company'))}-{slug(posting.get('title'))}"
    letter = draft_letter(profile, posting, match)
    notes = review_notes(profile, posting, match)
    prep = render_prep_markdown(build_prep(profile, posting, match))
    tailor = render_tailor_markdown(build_tailor(profile, posting, match))
    return packet, {
        "cover-letter.md": letter,
        "match.json": match_json(posting, match),
        "review-notes.md": notes,
        "interview-prep.md": prep,
        "tailored-resume.md": tailor,
    }


def _cmd_apply(args):
    profile, posting, code = _match_both(args)
    if code:
        return code
    match = match_profile(profile, posting)
    if (match["score"] or 0) < args.min_coverage:
        print(f"refusing to apply: score below --min-coverage {args.min_coverage}",
              file=sys.stderr)
        return 4
    deadline = None
    if args.deadline is not None:
        try:
            deadline = validate_deadline(args.deadline)
        except DeadlineError as exc:
            print(f"error: cannot apply: {exc}", file=sys.stderr)
            return 1
    source = None
    referrer = None
    if args.source is not None:
        try:
            cleaned = validate_source(args.source, args.referrer)
        except SourceError as exc:
            print(f"error: cannot apply: {exc}", file=sys.stderr)
            return 1
        source = cleaned["source"]
        referrer = cleaned["referrer"]
    packet, files = _build_packet(profile, posting, match, args.out_dir)
    log_path = args.log or str(Path(args.out_dir) / "applications.jsonl")
    if args.dry_run:
        print(f"(dry run: would write packet to {packet}/)")
        for name in files:
            print(f"  {packet}/{name}")
        print(f"(dry run: would log to {log_path})")
        print(match_report(posting, match), end="")
        return 0
    # Dedupe check happens before any file is written.
    existing = read_log(log_path)
    for entry in existing:
        if (entry["company"].strip().lower() == (posting.get("company") or "").strip().lower()
                and entry["title"].strip().lower() == (posting.get("title") or "").strip().lower()):
            print(f"error: already applied to {posting.get('company')!r} / "
                  f"{posting.get('title')!r} as {entry['id']} (status: {entry['status']})",
                  file=sys.stderr)
            return 5
    packet.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (packet / name).write_text(content, encoding="utf-8")
    try:
        entry = add_application(
            log_path,
            company=posting.get("company") or "unknown company",
            title=posting.get("title") or "unknown role",
            source=source,
            match_score=match["score"],
            packet_dir=str(packet),
            deadline=deadline,
            referrer=referrer,
            required_matched=match["required_matched"],
            required_missing=match["required_missing"],
            preferred_matched=match["preferred_matched"],
            preferred_missing=match["preferred_missing"],
        )
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 5
    print(f"packet written to {packet}/")
    print(f"logged as {entry['id']} (status: applied)")
    print("Review the letter before submitting — autojob does not submit applications.")
    return 0


def _cmd_list(args):
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.status:
        entries = [e for e in entries if e["status"] == args.status]
    if args.json:
        print(json.dumps(entries, indent=2, sort_keys=True))
    else:
        if not entries:
            print("no applications logged")
        for entry in entries:
            print(f"{entry['id']}  {entry['applied_at']}  {entry['company']} — "
                  f"{entry['title']}  [{entry['status']}]")
    return 0


def _cmd_status(args):
    try:
        entry = set_status(args.log, args.id, args.to)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 6
    print(f"{entry['id']}: {entry['company']} — {entry['title']} is now {entry['status']}")
    return 0


def _cmd_prep(args):
    profile, posting, code = _match_both(args)
    if code:
        return code
    prep = build_prep(profile, posting)
    if args.json:
        text = prep_json(prep)
    else:
        text = render_prep_markdown(prep)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_tailor(args):
    profile, posting, code = _match_both(args)
    if code:
        return code
    tailor = build_tailor(profile, posting)
    if args.json:
        text = tailor_json(tailor)
    else:
        text = render_tailor_markdown(tailor)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_followup(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            entry = record_followup(args.log, args.id,
                                    today=today.isoformat(), note=args.note)
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        print(f"recorded a follow-up for {entry['id']}: {entry['company']} — "
              f"{entry['title']} (sent {today.isoformat()})")
        return 0
    if args.id or args.note:
        print("error: --id and --note need --record", file=sys.stderr)
        return 1
    if args.wait_days is not None and args.wait_days < 0:
        print("error: --wait-days must be 0 or more", file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    name = None
    if args.profile:
        profile = _load_profile(args.profile)
        if profile is None:
            return 2
        name = profile["name"]
    items = due_followups(entries, today=today, wait_days=args.wait_days)
    if args.json:
        print(followup_json(items, name), end="")
    else:
        print(followup_report(items, name), end="")
    return 0


def _cmd_digest(args):
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.stale_days < 0:
        print("error: --stale-days must be 0 or more", file=sys.stderr)
        return 1
    if args.wait_days is not None and args.wait_days < 0:
        print("error: --wait-days must be 0 or more", file=sys.stderr)
        return 1
    data = build_digest(entries, today=today,
                        stale_days=args.stale_days, wait_days=args.wait_days)
    text = digest_json(data) if args.json else render_digest_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_plan(args):
    profile = _load_profile(args.profile)
    if profile is None:
        return 2
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    data = build_plan(profile, entries, today=today)
    text = plan_json(data) if args.json else render_plan_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_rank(args):
    profile = _load_profile(args.profile)
    if profile is None:
        return 2
    if args.min_score is not None and not 0.0 <= args.min_score <= 1.0:
        print("error: --min-score must be between 0 and 1, "
              f"got {args.min_score}", file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        data = rank_postings(profile, args.postings_dir, entries,
                             min_score=args.min_score,
                             include_applied=args.include_applied)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = rank_json(data) if args.json else render_rank_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_salary(args):
    profile = _load_profile(args.profile)
    if profile is None:
        return 2
    if args.min_pay is not None and args.min_pay < 0:
        print(f"error: --min-pay must be 0 or more, got {args.min_pay}",
              file=sys.stderr)
        return 1
    if args.min_pay is not None:
        # A floor only means something in the target's currency; without
        # one there is nothing to compare it against.
        target = target_from_profile(profile)
        if target is None or target.get("currency") is None:
            print("error: --min-pay needs a salary target with a currency "
                  "in the profile", file=sys.stderr)
            return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        data = analyze_salaries(profile, args.postings_dir, entries,
                                min_pay=args.min_pay)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = salary_json(data) if args.json else render_salary_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_offer(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    record_only = [args.base, args.currency, args.bonus, args.equity,
                   args.vesting_years, args.pto_days, args.mode,
                   args.start, args.deadline, args.note]
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            offer = validate_offer({
                "base": args.base,
                "currency": args.currency,
                "period": args.period,
                "bonus": args.bonus,
                "equity": args.equity,
                "vesting_years": args.vesting_years,
                "pto_days": args.pto_days,
                "work_mode": args.mode,
                "start_date": args.start,
                "deadline": args.deadline,
                "note": args.note,
            }, today=today.isoformat())
        except OfferError as exc:
            print(f"error: cannot record offer: {exc}", file=sys.stderr)
            return 1
        try:
            entry = record_offer(args.log, args.id, offer,
                                 today=today.isoformat())
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        print(f"recorded an offer for {entry['id']}: {entry['company']} — "
              f"{entry['title']} ({offer['currency']} "
              f"{offer['base_annual']:,.0f} base annual)")
        return 0
    if args.id or any(value is not None for value in record_only):
        print("error: --id, --base, --currency, --bonus, --equity, "
              "--vesting-years, --pto-days, --mode, --start, --deadline "
              "and --note need --record", file=sys.stderr)
        return 1
    profile = None
    if args.profile:
        profile = _load_profile(args.profile)
        if profile is None:
            return 2
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = compare_offers(entries, profile=profile, today=today)
    text = offer_json(data) if args.json else render_offer_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_interview(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    record_only = [args.date, args.round, args.interviewer, args.mode,
                   args.duration_minutes, args.time, args.notes]
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            interview = validate_interview({
                "date": args.date,
                "round": args.round,
                "interviewer": args.interviewer,
                "mode": args.mode,
                "duration_minutes": args.duration_minutes,
                "time": args.time,
                "notes": args.notes,
            }, today=today.isoformat())
        except InterviewError as exc:
            print(f"error: cannot record interview: {exc}", file=sys.stderr)
            return 1
        try:
            entry = record_interview(args.log, args.id, interview,
                                     today=today.isoformat())
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        print(f"recorded an interview for {entry['id']}: {entry['company']} — "
              f"{entry['title']} ({interview['round']} on {interview['date']})")
        return 0
    if any(value is not None for value in record_only):
        print("error: --date, --round, --interviewer, --mode, "
              "--duration-minutes, --time and --notes need --record", file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.id:
        entry, rounds = interviews_for(entries, args.id, today=today)
        if entry is None:
            print(f"error: no application with id {args.id!r}", file=sys.stderr)
            return 1
        upcoming = sorted((r for r in rounds if r["days_until"] is not None),
                          key=lambda r: (r["date"], r["round"] or ""))
        past = sorted((r for r in rounds if r["days_ago"] is not None),
                      key=lambda r: (r["date"], r["round"] or ""),
                      reverse=True)
        data = {"upcoming": upcoming, "past": past,
                "counts": {"upcoming": len(upcoming), "past": len(past),
                           "today": sum(1 for r in upcoming
                                        if r["days_until"] == 0)}}
    else:
        data = collect_interviews(entries, today=today)
    text = interview_json(data) if args.json else render_interview_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_contacts(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.quiet_days is not None and args.quiet_days < 0:
        print("error: --quiet-days must be 0 or more", file=sys.stderr)
        return 1
    record_only = [args.name, args.role, args.channel, args.date, args.note]
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            contact = validate_contact({
                "name": args.name,
                "role": args.role,
                "channel": args.channel,
                "date": args.date,
                "note": args.note,
            }, today=today.isoformat())
        except ContactError as exc:
            print(f"error: cannot record contact: {exc}", file=sys.stderr)
            return 1
        try:
            entry = record_contact(args.log, args.id, contact,
                                   today=today.isoformat())
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        print(f"recorded a contact for {entry['id']}: {entry['company']} — "
              f"{entry['title']} ({contact['name']}"
              + (f", {contact['role']}" if contact.get("role") else "")
              + f" on {contact['date']})")
        return 0
    if any(value is not None for value in record_only):
        print("error: --name, --role, --channel, --date and --note need "
              "--record", file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.id:
        entry, rows = contacts_for(entries, args.id, today=today,
                                   quiet_days=args.quiet_days)
        if entry is None:
            print(f"error: no application with id {args.id!r}", file=sys.stderr)
            return 1
        data = {"contacts": rows,
                "counts": {"total": len(rows),
                           "quiet": sum(1 for r in rows if r["quiet"])}}
    else:
        data = collect_contacts(entries, today=today,
                                quiet_days=args.quiet_days)
    text = contacts_json(data) if args.json else render_contacts_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_timeline(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    since = until = None
    for flag, value in (("--since", args.since), ("--until", args.until)):
        if value is not None:
            try:
                parsed = datetime.date.fromisoformat(value)
            except ValueError:
                print(f"error: {flag} must be YYYY-MM-DD, got {value!r}",
                      file=sys.stderr)
                return 1
            if flag == "--since":
                since = parsed
            else:
                until = parsed
    if since is not None and until is not None and since > until:
        print(f"error: --since {args.since} is after --until {args.until}",
              file=sys.stderr)
        return 1
    kinds = None
    if args.kind:
        bad = [kind for kind in args.kind if kind not in KINDS]
        if bad:
            print(f"error: unknown --kind {bad[0]!r}; one of: "
                  f"{', '.join(KINDS)}", file=sys.stderr)
            return 1
        kinds = args.kind
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.id and find_application(entries, args.id) is None:
        print(f"error: no application with id {args.id!r}", file=sys.stderr)
        return 1
    data = collect_timeline(entries, today=today, since=since, until=until,
                            kinds=kinds, app_id=args.id, oldest=args.oldest)
    text = timeline_json(data) if args.json else render_timeline_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_deadlines(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            deadline = validate_deadline(args.date)
        except DeadlineError as exc:
            print(f"error: cannot record closing date: {exc}",
                  file=sys.stderr)
            return 1
        try:
            entry = record_deadline(args.log, args.id, deadline,
                                    today=today.isoformat())
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        print(f"recorded a closing date for {entry['id']}: "
              f"{entry['company']} — {entry['title']} "
              f"(closes {deadline})")
        return 0
    if args.id or args.date:
        print("error: --id and --date need --record", file=sys.stderr)
        return 1
    if args.days is not None and args.days < 0:
        print(f"error: --days must be 0 or more, got {args.days}",
              file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = collect_deadlines(entries, today=today, days=args.days,
                             overdue_only=args.overdue)
    text = deadlines_json(data) if args.json else render_deadlines_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_goals(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    goals_path = args.goals or default_goals_path(args.log)
    if args.set_weekly is not None:
        try:
            data = set_weekly_target(goals_path, args.set_weekly)
        except GoalsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if data["weekly_target"] is None:
            print(f"weekly target cleared (was stored in {goals_path})")
        else:
            print(f"weekly target set to {data['weekly_target']} "
                  f"application"
                  + ("s" if data["weekly_target"] != 1 else "")
                  + f" a week (stored in {goals_path})")
        return 0
    if args.weeks is not None and args.weeks < 1:
        print(f"error: --weeks must be 1 or more, got {args.weeks}",
              file=sys.stderr)
        return 1
    try:
        goals = load_goals(goals_path)
    except GoalsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = collect_goals(entries, today=today,
                         weeks=args.weeks if args.weeks is not None else 8,
                         weekly_target=goals["weekly_target"])
    text = goals_json(data) if args.json else render_goals_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_rejections(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            rejection = validate_rejection(args.reason, args.note)
        except RejectionError as exc:
            print(f"error: cannot record rejection reason: {exc}",
                  file=sys.stderr)
            return 1
        try:
            entry = record_rejection(args.log, args.id, rejection,
                                     today=today.isoformat())
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        print(f"recorded a rejection reason for {entry['id']}: "
              f"{entry['company']} — {entry['title']} "
              f"({rejection['category']})")
        return 0
    if args.id or args.note:
        print("error: --id and --note need --record", file=sys.stderr)
        return 1
    if args.reason is not None and args.reason not in REJECTION_REASONS:
        print(f"error: --reason must be one of "
              f"{', '.join(REJECTION_REASONS)}, got {args.reason!r}",
              file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = collect_rejections(entries, today=today, reason=args.reason)
    text = (rejections_json(data) if args.json
            else render_rejections_markdown(data))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_sources(args):
    if args.record:
        if not args.id:
            print("error: --record needs --id APP-001", file=sys.stderr)
            return 1
        try:
            source = validate_source(args.source, args.referrer)
        except SourceError as exc:
            print(f"error: cannot record source: {exc}",
                  file=sys.stderr)
            return 1
        try:
            entry = record_source(args.log, args.id, source)
        except TrackerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 6
        detail = f" ({source['referrer']})" if source["referrer"] else ""
        print(f"recorded a source for {entry['id']}: "
              f"{entry['company']} — {entry['title']} "
              f"({source['source']}{detail})")
        return 0
    if args.id or args.referrer:
        print("error: --id and --referrer need --record", file=sys.stderr)
        return 1
    if args.source is not None and args.source not in SOURCE_CHOICES:
        print(f"error: --source must be one of "
              f"{', '.join(SOURCE_CHOICES)}, got {args.source!r}",
              file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = collect_sources(entries, source=args.source)
    text = (sources_json(data) if args.json
            else render_sources_markdown(data))
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_next(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.days is not None and args.days < 0:
        print(f"error: --days must be 0 or more, got {args.days}",
              file=sys.stderr)
        return 1
    for name, value in (("wait-days", args.wait_days),
                        ("quiet-days", args.quiet_days),
                        ("stale-days", args.stale_days)):
        if value is not None and value < 0:
            print(f"error: --{name} must be 0 or more, got {value}",
                  file=sys.stderr)
            return 1
    if args.kind:
        unknown = [kind for kind in args.kind if kind not in NEXT_KINDS]
        if unknown:
            print(f"error: --kind must be one of "
                  f"{', '.join(NEXT_KINDS)}, got {', '.join(unknown)}",
                  file=sys.stderr)
            return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        data = collect_next(entries, today=today,
                            days=args.days if args.days is not None else 7,
                            kinds=args.kind,
                            wait_days=args.wait_days,
                            quiet_days=(args.quiet_days
                                      if args.quiet_days is not None else 14),
                            stale_days=(args.stale_days
                                      if args.stale_days is not None else 14))
    except NextError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = next_json(data) if args.json else render_next_markdown(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_calendar(args):
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.days < 0:
        print(f"error: --days must be 0 or more, got {args.days}",
              file=sys.stderr)
        return 1
    if args.wait_days is not None and args.wait_days < 0:
        print("error: --wait-days must be 0 or more", file=sys.stderr)
        return 1
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    data = build_calendar(entries, today=today, days=args.days,
                          wait_days=args.wait_days)
    text = calendar_json(data) if args.json else render_ics(data)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def _cmd_report(args):
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.today is not None:
        try:
            today = datetime.date.fromisoformat(args.today)
        except ValueError:
            print(f"error: --today must be YYYY-MM-DD, got {args.today!r}",
                  file=sys.stderr)
            return 1
    else:
        today = datetime.date.today()
    if args.stale_days < 0:
        print("error: --stale-days must be 0 or more", file=sys.stderr)
        return 1
    data = compute_analytics(entries, today=today, stale_days=args.stale_days)
    if args.json:
        print(analytics_json(data), end="")
    else:
        print(analytics_report(data), end="")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="autojob",
        description="Job-application assistant that prepares packets from your "
                    "real profile — never invents experience, never submits.",
    )
    parser.add_argument("--version", action="version", version=f"autojob {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check a profile file")
    p.add_argument("--profile", required=True)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("parse", help="show the structured posting")
    p.add_argument("--posting", required=True)
    p.add_argument("--profile")
    p.set_defaults(func=_cmd_parse)

    p = sub.add_parser("match", help="score a profile against a posting")
    p.add_argument("--profile", required=True)
    p.add_argument("--posting", required=True)
    p.add_argument("--fail-under", type=float, default=None)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_match)

    p = sub.add_parser("draft", help="draft a cover letter from profile facts only")
    p.add_argument("--profile", required=True)
    p.add_argument("--posting", required=True)
    p.add_argument("--min-coverage", type=float, default=0.0)
    p.add_argument("--out")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_draft)

    p = sub.add_parser("apply", help="build an application packet and log it")
    p.add_argument("--profile", required=True)
    p.add_argument("--posting", required=True)
    p.add_argument("--min-coverage", type=float, default=0.0)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--log")
    p.add_argument("--source")
    p.add_argument("--referrer", default=None,
                   help="with --source referral: who referred you; with "
                        "any other source: the recruiter or person who "
                        "brought the role, if any")
    p.add_argument("--deadline", default=None,
                   help="the posting's closing date YYYY-MM-DD, recorded "
                        "on the application")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=_cmd_apply)

    p = sub.add_parser("list", help="list logged applications")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--status")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("status", help="move an application to a new status")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--id", required=True)
    p.add_argument("--to", required=True)
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("prep", help="build an interview-prep packet from profile facts")
    p.add_argument("--profile", required=True)
    p.add_argument("--posting", required=True)
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_prep)

    p = sub.add_parser("tailor", help="build a tailored resume section from "
                                       "profile facts for one posting")
    p.add_argument("--profile", required=True)
    p.add_argument("--posting", required=True)
    p.add_argument("--out")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_tailor)

    p = sub.add_parser("followup", help="list applications due a follow-up, "
                                         "or record one as sent")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--profile",
                   help="profile file: supplies the name for draft emails")
    p.add_argument("--wait-days", type=int, default=None,
                   help="override the per-stage wait for every stage (default: "
                        "7 applied/screening, 3 interview, 2 offer)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for")
    p.add_argument("--record", action="store_true",
                   help="record a follow-up as sent instead of listing due ones")
    p.add_argument("--note", default=None,
                   help="with --record: a short note stored with the follow-up")
    p.set_defaults(func=_cmd_followup)

    p = sub.add_parser("digest", help="one-page digest over the whole log: "
                                       "funnel, stale, due follow-ups, and "
                                       "the skills postings keep asking for")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the digest to this file instead of stdout")
    p.add_argument("--stale-days", type=int, default=14,
                   help="flag applications with no movement in this many days (default 14)")
    p.add_argument("--wait-days", type=int, default=None,
                   help="override the per-stage follow-up wait for every stage")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.set_defaults(func=_cmd_digest)

    p = sub.add_parser("plan", help="gap-closing plan over the log: skills "
                                      "you have but cannot prove, skills to "
                                      "learn, and the quick wins")
    p.add_argument("--profile", required=True)
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the plan to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("rank", help="rank a folder of postings by fit: "
                                     "best application first, already-applied "
                                     "postings left out unless asked for")
    p.add_argument("--profile", required=True)
    p.add_argument("--postings-dir", required=True,
                   help="directory of .md / .txt postings to rank")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--min-score", type=float, default=None,
                   help="drop postings scoring below this (0-1)")
    p.add_argument("--include-applied", action="store_true",
                   help="also rank postings already in the log")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the ranking to this file instead of stdout")
    p.set_defaults(func=_cmd_rank)

    p = sub.add_parser("salary", help="compare what a folder of postings "
                                       "pays against the profile's salary "
                                       "target, with per-currency medians")
    p.add_argument("--profile", required=True)
    p.add_argument("--postings-dir", required=True,
                   help="directory of .md / .txt postings to analyse")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--min-pay", type=float, default=None,
                   help="drop postings whose max is below this, in the "
                        "target's currency (needs a salary target with "
                        "a currency in the profile)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the analysis to this file instead of stdout")
    p.set_defaults(func=_cmd_salary)

    p = sub.add_parser("offer", help="compare recorded offers against the "
                                       "profile's salary target, or record "
                                       "one with --record")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--profile", default=None,
                   help="profile file: supplies the salary target for "
                        "verdicts and ranking (without it, offers are "
                        "listed, not compared)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the comparison to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD for deadlines (default: today)")
    p.add_argument("--record", action="store_true",
                   help="record an offer instead of comparing")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for")
    p.add_argument("--base", type=float, default=None,
                   help="with --record: annual base (or per period, see --period)")
    p.add_argument("--currency", default=None,
                   help="with --record: 3-letter currency code, e.g. CAD")
    p.add_argument("--period", default=None,
                   choices=["annual", "hourly", "monthly"],
                   help="with --record: how --base is stated "
                        "(default: annual; hourly annualises at 2,080 hours)")
    p.add_argument("--bonus", type=float, default=None,
                   help="with --record: annual bonus (assumed annual)")
    p.add_argument("--equity", type=float, default=None,
                   help="with --record: equity value; counts towards the "
                        "first-year total only with --vesting-years")
    p.add_argument("--vesting-years", type=int, default=None,
                   help="with --record: years the equity vests over")
    p.add_argument("--pto-days", type=int, default=None,
                   help="with --record: paid time off days")
    p.add_argument("--mode", default=None,
                   choices=["remote", "hybrid", "onsite"],
                   help="with --record: work mode of the offer")
    p.add_argument("--start", default=None,
                   help="with --record: start date YYYY-MM-DD")
    p.add_argument("--deadline", default=None,
                   help="with --record: decision deadline YYYY-MM-DD")
    p.add_argument("--note", default=None,
                   help="with --record: a short note stored with the offer")
    p.set_defaults(func=_cmd_offer)

    p = sub.add_parser("interview", help="list recorded interview rounds, "
                                          "or record one with --record")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the schedule to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--record", action="store_true",
                   help="record an interview instead of listing")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for; "
                        "without it: show only this application's rounds")
    p.add_argument("--date", default=None,
                   help="with --record: interview date YYYY-MM-DD")
    p.add_argument("--round", default=None,
                   help="with --record: round label, e.g. 'Hiring manager'")
    p.add_argument("--interviewer", default=None,
                   help="with --record: interviewer name")
    p.add_argument("--mode", default=None,
                   choices=["phone", "video", "onsite"],
                   help="with --record: how the interview happens")
    p.add_argument("--duration-minutes", type=int, default=None,
                   help="with --record: scheduled length in minutes")
    p.add_argument("--time", default=None,
                   help="with --record: start time HH:MM (24-hour); a round "
                        "with a time exports as a timed calendar event")
    p.add_argument("--note", dest="notes", default=None,
                   help="with --record: a short note stored with the round")
    p.set_defaults(func=_cmd_interview)

    p = sub.add_parser("calendar", help="export the next N days as an .ics "
                                         "calendar: interview rounds, offer "
                                         "decision deadlines, posting "
                                         "closing dates, and follow-ups "
                                         "due today")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--days", type=int, default=14,
                   help="how many days ahead to include (default 14; "
                        "0 means today only)")
    p.add_argument("--wait-days", type=int, default=None,
                   help="override the per-stage follow-up wait for every stage")
    p.add_argument("--json", action="store_true",
                   help="print the events as JSON instead of .ics")
    p.add_argument("--out", default=None,
                   help="write the .ics file to this path instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.set_defaults(func=_cmd_calendar)

    p = sub.add_parser("contacts", help="list the people behind each "
                                         "application, quietest first, or "
                                         "record one with --record")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the contacts list to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--quiet-days", type=int, default=None,
                   help="a contact is quiet after this many days without "
                        "speaking to them (default 14; exactly N days is "
                        "not quiet yet)")
    p.add_argument("--record", action="store_true",
                   help="record a contact instead of listing")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for; "
                        "without it: show only this application's contacts")
    p.add_argument("--name", default=None,
                   help="with --record: the person's name")
    p.add_argument("--role", default=None,
                   help="with --record: their role, e.g. 'Recruiter'")
    p.add_argument("--channel", default=None,
                   choices=["email", "linkedin", "phone", "in-person",
                            "other"],
                   help="with --record: how you reached them")
    p.add_argument("--date", default=None,
                   help="with --record: date you first reached them, or "
                        "last spoke to them (YYYY-MM-DD, required)")
    p.add_argument("--note", default=None,
                   help="with --record: a short note stored with the contact")
    p.set_defaults(func=_cmd_contacts)

    p = sub.add_parser("timeline", help="one chronological feed of "
                                          "everything recorded — "
                                          "applications, status moves, "
                                          "follow-ups, interviews, contacts, "
                                          "offers — newest first")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the timeline to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--id", default=None,
                   help="show only this application's events")
    p.add_argument("--since", default=None,
                   help="first event date to include, YYYY-MM-DD (inclusive)")
    p.add_argument("--until", default=None,
                   help="last event date to include, YYYY-MM-DD (inclusive)")
    p.add_argument("--kind", action="append", default=None,
                   help="only events of this kind; repeatable "
                        f"({', '.join(KINDS)})")
    p.add_argument("--oldest", action="store_true",
                   help="chronological order instead of newest first")
    p.set_defaults(func=_cmd_timeline)

    p = sub.add_parser("deadlines", help="list posting closing dates, "
                                          "soonest first, or record one "
                                          "with --record")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the list to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--days", type=int, default=None,
                   help="only deadlines this many days out or sooner "
                        "(overdue ones are always shown)")
    p.add_argument("--overdue", action="store_true",
                   help="only deadlines that already passed")
    p.add_argument("--record", action="store_true",
                   help="record a closing date instead of listing")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for")
    p.add_argument("--date", default=None,
                   help="with --record: the closing date YYYY-MM-DD")
    p.set_defaults(func=_cmd_deadlines)

    p = sub.add_parser("goals", help="weekly application target and "
                                       "streak: set one with --set-weekly")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--goals", default=None,
                   help="goals file (default: goals.json beside the log)")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the goals view to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--set-weekly", type=int, default=None,
                   help="set the weekly application target (0 clears it)")
    p.add_argument("--weeks", type=int, default=None,
                   help="how many weeks back the table runs (default 8)")
    p.set_defaults(func=_cmd_goals)

    p = sub.add_parser("rejections", help="list rejected applications and "
                                            "why, or record a reason with "
                                            "--record")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the list to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--reason", default=None,
                   help="only rejections with this reason (list mode), or "
                        "the reason to record (with --record): "
                        + ", ".join(REJECTION_REASONS))
    p.add_argument("--record", action="store_true",
                   help="record a rejection reason instead of listing")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for")
    p.add_argument("--note", default=None,
                   help="with --record: what you were told, in your own words")
    p.set_defaults(func=_cmd_rejections)

    p = sub.add_parser("sources", help="which channels your applications "
                                         "came through and what worked, or "
                                         "record a source with --record")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the list to this file instead of stdout")
    p.add_argument("--source", default=None,
                   help="only this channel (list mode), or the channel to "
                        "record (with --record): "
                        + ", ".join(SOURCE_CHOICES))
    p.add_argument("--referrer", default=None,
                   help="with --record: who referred you (required for a "
                        "referral), or the recruiter or person who brought "
                        "the role")
    p.add_argument("--record", action="store_true",
                   help="record a source instead of listing")
    p.add_argument("--id", default=None,
                   help="with --record: the application to record for")
    p.set_defaults(func=_cmd_sources)

    p = sub.add_parser("next", help="one prioritised list of what to do: "
                                      "due follow-ups, upcoming interviews, "
                                      "closing dates, offer deadlines, quiet "
                                      "contacts, stale applications, and "
                                      "rejections missing a reason")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None,
                   help="write the list to this file instead of stdout")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD (default: today)")
    p.add_argument("--days", type=int, default=7,
                   help="how far ahead dated actions are looked for "
                        "(overdue ones are always shown; default 7)")
    p.add_argument("--kind", action="append", default=None,
                   help="only this kind of action (repeatable): "
                        + ", ".join(NEXT_KINDS))
    p.add_argument("--wait-days", type=int, default=None,
                   help="follow-up wait in days (default: per-stage waits)")
    p.add_argument("--quiet-days", type=int, default=14,
                   help="days of silence before a contact counts as quiet "
                        "(default 14)")
    p.add_argument("--stale-days", type=int, default=14,
                   help="days without movement before an application "
                        "counts as stale (default 14)")
    p.set_defaults(func=_cmd_next)

    p = sub.add_parser("report", help="funnel analytics over logged applications")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
    p.add_argument("--stale-days", type=int, default=14,
                   help="flag applications with no movement in this many days (default 14)")
    p.add_argument("--today", default=None,
                   help="reference date YYYY-MM-DD for stale checks (default: today)")
    p.set_defaults(func=_cmd_report)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
