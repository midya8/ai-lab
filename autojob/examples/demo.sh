#!/usr/bin/env bash
# autojob demo: validate, match, draft, apply, list, report — all offline,
# all against the fictional example profile and posting.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

python3 -m autojob validate --profile examples/profile.example.json
python3 -m autojob match --profile examples/profile.example.json \
    --posting examples/job.example.md
python3 -m autojob draft --profile examples/profile.example.json \
    --posting examples/job.example.md --dry-run | head -20
python3 -m autojob apply --profile examples/profile.example.json \
    --posting examples/job.example.md --out-dir "$WORK/applications" \
    --deadline "$(date -d '+10 days' +%F)"
python3 -m autojob apply --profile examples/profile.example.json \
    --posting examples/job2.example.md --out-dir "$WORK/applications" \
    --deadline "$(date -d '+30 days' +%F)"
python3 -m autojob list --log "$WORK/applications/applications.jsonl"
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --to screening
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-002 --to screening
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-002 --to interview
python3 -m autojob report --log "$WORK/applications/applications.jsonl"
python3 -m autojob prep --profile examples/profile.example.json \
    --posting examples/job.example.md --out "$WORK/prep.md"
head -12 "$WORK/prep.md"
python3 -m autojob tailor --profile examples/profile.example.json \
    --posting examples/job.example.md --out "$WORK/tailored.md"
head -12 "$WORK/tailored.md"
# Follow-ups: nothing is due yet (both applied today), but 8 days on both
# are past the 7-day applied/screening wait. Record one as sent, and show
# the other is still due.
FUTURE="$(date -d '+8 days' +%F)"
python3 -m autojob followup --log "$WORK/applications/applications.jsonl" \
    --profile examples/profile.example.json --today "$FUTURE"
python3 -m autojob followup --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --today "$FUTURE"
python3 -m autojob followup --log "$WORK/applications/applications.jsonl" \
    --profile examples/profile.example.json --today "$FUTURE"
# Digest: one page over the whole log — funnel, stale, due follow-ups,
# and the skills both postings keep asking for that the profile lacks.
python3 -m autojob digest --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
# Plan: turn the gaps into work — the skills the profile has but no
# bullet proves, the skills to learn, and the three quick wins.
python3 -m autojob plan --profile examples/profile.example.json \
    --log "$WORK/applications/applications.jsonl" --today "$FUTURE"
# Rank: all three example postings by fit — Northwind first, Harbour
# second, Vector Labs last (the profile has Python but none of the ML
# stack). The two already-applied postings are excluded by default, so
# the plain rank names Vector Labs as the only one left to apply to.
python3 -m autojob rank --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl" \
    --include-applied
python3 -m autojob rank --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl"
# Salary: all three example postings list CAD ranges; the profile target
# is CAD 95,000–120,000. Vector Labs pays above it (130,000–160,000),
# Northwind and Harbour meet it; the CAD medians are 105,000–130,000
# (midpoint 117,500). The two applied postings are flagged with their ids.
python3 -m autojob salary --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl"
# Offers: Northwind offers CAD 130,000 base + 10,000 bonus + 20,000 equity
# over 4 years (first year 145,000 — above the 95,000–120,000 target on
# base alone), Harbour offers CAD 105,000 + 5,000 bonus (first year
# 110,000, meets the target). Both deadlines fall 14 days out, so 8 days
# on the comparison shows 6 days left to decide; Northwind ranks first.
OFFER_DEADLINE="$(date -d '+14 days' +%F)"
python3 -m autojob offer --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --base 130000 --currency CAD --bonus 10000 \
    --equity 20000 --vesting-years 4 --pto-days 20 --mode remote \
    --deadline "$OFFER_DEADLINE" --today "$FUTURE"
python3 -m autojob offer --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --base 105000 --currency CAD --bonus 5000 \
    --mode hybrid --deadline "$OFFER_DEADLINE" --today "$FUTURE"
python3 -m autojob offer --profile examples/profile.example.json \
    --log "$WORK/applications/applications.jsonl" --today "$FUTURE"
# Interviews: Harbour is at interview stage, so record its rounds — a
# recruiter screen a day before the reference date (past) and a
# hiring-manager round two days after it (upcoming, video, 45 minutes).
# The schedule shows both sides of today.
PAST_DAY="$(date -d '+7 days' +%F)"
INTERVIEW_DAY="$(date -d '+10 days' +%F)"
python3 -m autojob interview --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --date "$PAST_DAY" --round "Recruiter screen" \
    --mode phone --duration-minutes 30 --today "$FUTURE"
python3 -m autojob interview --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --date "$INTERVIEW_DAY" --round "Hiring manager" \
    --interviewer "Priya Nair" --mode video --duration-minutes 45 \
    --time 14:30 --note "Bring the dashboard story." --today "$FUTURE"
python3 -m autojob interview --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
# Calendar: export the next 14 days as .ics — the hiring-manager round
# (timed, 14:30 for 45 minutes), the Northwind posting closing date
# (10 days out, inside the window), and both offer deadlines (14 days
# out, inside the window). The Harbour closing date is 30 days out, so
# it stays out of the file. No follow-up is due: both applications had
# activity recorded today (the follow-up and the offers), so the file
# holds 4 events.
python3 -m autojob calendar --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --out "$WORK/search.ics"
grep -c "BEGIN:VEVENT" "$WORK/search.ics"
# Contacts: the people behind the applications — a recruiter at Northwind
# you spoke to today (in demo time, 8 days after applying), and a hiring
# manager at Harbour you emailed the day you applied. Eight days on, the
# Harbour contact has gone quiet by a 7-day bar; Northwind has not.
python3 -m autojob contacts --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --name "Sam Chen" --role Recruiter \
    --channel linkedin --date "$FUTURE" --today "$FUTURE"
python3 -m autojob contacts --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --name "Alex Rivera" --role "Hiring manager" \
    --channel email --date "$(date +%F)" --today "$FUTURE"
python3 -m autojob contacts --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob contacts --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --quiet-days 7
# Deadlines: both postings were applied to with a closing date — Northwind
# 10 days out, Harbour 30 days out. Eight days on, Northwind closes in
# 2 days and Harbour in 22; nothing is overdue yet. The Harbour listing
# is then updated to close 25 days out, and the list shows 17 days left.
python3 -m autojob deadlines --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob deadlines --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --overdue
python3 -m autojob deadlines --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --date "$(date -d '+25 days' +%F)" \
    --today "$FUTURE"
python3 -m autojob deadlines --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
# Timeline: everything recorded so far in one feed, newest first — both
# applications, the status moves, the follow-up, the offers, the
# interviews, and the contacts. Eight days on, the feed runs from today
# to the hiring-manager round two days out.
python3 -m autojob timeline --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob timeline --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --kind interview --kind offer --oldest
# Goals: set a weekly target of 2 applications. Both were filed this
# week, so the target is met and the streak is 1. Eight days on, the
# new week has none yet, so the week shows 0 of 2 with 2 to go — but
# the streak still stands at 1, because last week met the target.
python3 -m autojob goals --log "$WORK/applications/applications.jsonl" \
    --set-weekly 2
python3 -m autojob goals --log "$WORK/applications/applications.jsonl"
python3 -m autojob goals --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
echo "demo OK: packets in $WORK/applications (removed on exit)"
