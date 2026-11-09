"""autojob CLI.

Subcommands (all offline, stdlib only):

  validate  --profile P              check a profile file
  parse     --posting J [--profile P] show the structured posting
  match     --profile P --posting J [--fail-under X] [--json]
  draft     --profile P --posting J [--min-coverage X] [--out F] [--dry-run]
  apply     --profile P --posting J [--min-coverage X] [--out-dir D] [--log L] [--dry-run]
  list      [--log L] [--status S]
  status    --log L --id APP-001 --to interview
  report    [--log L] [--json]

Exit codes: 0 ok · 1 usage error · 2 profile/posting load error ·
3 match below --fail-under · 4 draft/apply refused below --min-coverage ·
5 duplicate application · 6 illegal status transition.

autojob prepares application packets for a human to review and submit. It
never submits anything and never invents experience: every fact in a draft
comes from the profile file, and gaps are reported, not filled in.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .letter import draft_letter, review_notes
from .match import match_profile
from .posting import parse_posting
from .profile import ProfileError, load_profile, skill_map
from .report import log_report, match_json, match_report
from .tracker import (TrackerError, add_application, read_log, set_status, slug)

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
    return packet, {
        "cover-letter.md": letter,
        "match.json": match_json(posting, match),
        "review-notes.md": notes,
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
            source=args.source,
            match_score=match["score"],
            packet_dir=str(packet),
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


def _cmd_report(args):
    try:
        entries = read_log(args.log)
    except TrackerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"total": len(entries), "entries": entries},
                         indent=2, sort_keys=True))
    else:
        print(log_report(entries), end="")
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

    p = sub.add_parser("report", help="summary of logged applications")
    p.add_argument("--log", default=f"{DEFAULT_OUT_DIR}/applications.jsonl")
    p.add_argument("--json", action="store_true")
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
