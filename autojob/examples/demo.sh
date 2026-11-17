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
    --deadline "$(date -d '+10 days' +%F)" \
    --resume "resume-data-analyst-v3.pdf"
python3 -m autojob apply --profile examples/profile.example.json \
    --posting examples/job2.example.md --out-dir "$WORK/applications" \
    --deadline "$(date -d '+30 days' +%F)" --source job-board
# Sources: Harbour came through a job board (recorded at apply time);
# Northwind came through a referral — Maya Patel — recorded afterwards.
# Eight days on, the job-board row shows 1 of 1 reaching an interview
# and the referral row 0 of 1, with Maya Patel listed to follow up with.
python3 -m autojob sources --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --source referral --referrer "Maya Patel"
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
# Negotiation brief: what the Northwind offer is worth and what to say
# about it. Eight days on, the offer's first year is 145,000 — above
# the 95,000–120,000 target on base alone, ranked 1 of the 2 comparable
# offers, with the posting's own range (95,000–120,000) meeting the
# target and 6 days left to decide. The strengths to cite start with
# SQL (4 yrs, advanced) with a verbatim bullet; Communication, dbt,
# Snowflake, and A/B Testing are listed as do-not-claim. Harbour is
# still in play at interview stage with its own offer (first year
# 110,000), so it shows up as the other live option. The brief
# suggests no number — it states the facts and leaves the ask to you.
python3 -m autojob negotiate --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --today "$FUTURE"
python3 -m autojob negotiate --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --today "$FUTURE" --json --out "$WORK/negotiate.json"
grep '"rank": 1' "$WORK/negotiate.json"
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
# Debriefs: write up a round while it is fresh — what went well, what to
# improve, and the questions they actually asked. The recruiter screen
# was yesterday, so it gets a debrief (rated 4 of 5); the hiring-manager
# round is still two days out, so there is nothing to write up yet.
python3 -m autojob debrief --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --date "$PAST_DAY" --round "Recruiter screen" \
    --rating 4 --went-well "Dashboard story landed; SQL answers were quick." \
    --improve "Tighter answer on stakeholder pushback." \
    --question "Walk me through your most recent project." \
    --question "How do you handle a stakeholder who disagrees with the numbers?" \
    --today "$FUTURE"
python3 -m autojob debrief --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
# Question bank: the questions the debriefs recorded, most-asked first.
# Eight days on there are two, neither with a prepared answer yet; store
# one for the walk-through question and the --unanswered view names just
# the stakeholder one.
python3 -m autojob questions --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob questions --log "$WORK/applications/applications.jsonl" \
    --record --question "Walk me through your most recent project." \
    --answer "Most recent role first: the Data Analyst work, then the reporting work before it." \
    --today "$FUTURE"
python3 -m autojob questions --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob questions --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --unanswered
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
# Notes: what you learn along the way, written down dated — the team
# rebuilding the reporting platform at Northwind, the two-week ship
# cadence at Harbour. Eight days on, both notes are on the list with
# their tags; the --tag view names just the company one; and "meetup"
# finds Northwind through the note, like any other recorded fact.
python3 -m autojob notes --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --date "$FUTURE" \
    --text "Met Sam Chen at the Toronto data meetup — the team is rebuilding the reporting platform." \
    --tag company --today "$FUTURE"
python3 -m autojob notes --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --date "$FUTURE" \
    --text "Priya mentioned the team ships every two weeks; ask about the on-call rotation." \
    --tag interview --today "$FUTURE"
python3 -m autojob notes --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob notes --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --tag company
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "meetup"
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
# Debriefs, 8 days further on: the hiring-manager round is 6 days past
# with no write-up, so the list flags it as missing.
python3 -m autojob debrief --log "$WORK/applications/applications.jsonl" \
    --today "$(date -d '+16 days' +%F)"
# Write it up now that it happened — the SQL question came up, and the
# Power BI migration did not — then review what both Harbour rounds
# actually tested against the posting. One of four required skills came
# up (SQL, in the hiring-manager round); Power BI, Python, and
# Communication never did, and neither did any preferred skill. Prep
# called one of its three skill questions (SQL). The other two questions
# were off-script — the walk-through and the stakeholder one name no
# skill the posting asked for. Northwind has no debrief, so it is
# skipped.
python3 -m autojob debrief --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --date "$INTERVIEW_DAY" --round "Hiring manager" \
    --rating 4 --went-well "Dashboard story landed; the SQL question was straightforward." \
    --improve "Tighter answer on the Power BI migration." \
    --question "Tell me about a time you used SQL to answer a business question." \
    --today "$(date -d '+16 days' +%F)"
python3 -m autojob review --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl"
# Goals: set a weekly target of 2 applications. Both were filed this
# week, so the target is met and the streak is 1. Eight days on, the
# new week has none yet, so the week shows 0 of 2 with 2 to go — but
# the streak still stands at 1, because last week met the target.
python3 -m autojob goals --log "$WORK/applications/applications.jsonl" \
    --set-weekly 2
python3 -m autojob goals --log "$WORK/applications/applications.jsonl"
python3 -m autojob goals --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
# Resumes: record which file an application got while it is still
# live — recording is refused once an application is terminal, the
# same rule as a closing date or a source. Harbour went out with the
# v2 resume, the older one before the summary was shortened; Northwind
# already carries the v3, recorded at apply time.
python3 -m autojob resume --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --file "resume-data-analyst-v2.pdf" \
    --note "the older one, before the summary was shortened"
# Rejections: the funnel ends somewhere — record why. Harbour reached
# the interview stage and then said no over salary; Vector Labs never
# got past the application, and the gap was the ML stack the profile
# does not have (PyTorch, TensorFlow, Statistics, Machine Learning).
# Both were applied to today and rejected today, so the report shows
# 0 days to a decision for each, 8 days ago in demo time; the
# missing-skills roll-up names what the rejected postings kept asking
# for, and the --reason filter shows just the skills-gap row.
python3 -m autojob apply --profile examples/profile.example.json \
    --posting examples/job3.example.md --out-dir "$WORK/applications" \
    --source company-site
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-002 --to rejected
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-003 --to rejected
python3 -m autojob rejections --log "$WORK/applications/applications.jsonl" \
    --record --id APP-002 --reason salary \
    --note "Their band topped out below the target range." --today "$FUTURE"
python3 -m autojob rejections --log "$WORK/applications/applications.jsonl" \
    --record --id APP-003 --reason skills-gap \
    --note "Wanted PyTorch and TensorFlow; the profile has neither." \
    --today "$FUTURE"
python3 -m autojob rejections --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob rejections --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --reason skills-gap
# Sources: three channels, one application each — the job board led
# somewhere (Harbour reached an interview), the referral and the
# company site have not yet. The referral filter shows just Maya Patel.
python3 -m autojob sources --log "$WORK/applications/applications.jsonl"
python3 -m autojob sources --log "$WORK/applications/applications.jsonl" \
    --source referral
# Next actions: one prioritised list of what to do. Eight days on,
# the Northwind posting closes in 2 days and its offer decision is
# due in 6 — two actions, soonest first. Eight days further on, the
# closing date is 6 days overdue, the offer decision 2 days overdue,
# and the Northwind application has gone 8 days without a word, past
# the 7-day screening wait, so a follow-up is due too. Sam Chen has
# also gone quiet by a 7-day bar (not by the default 14), and the
# --kind filter shows just the two dated actions.
python3 -m autojob next --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
LATER="$(date -d '+16 days' +%F)"
python3 -m autojob next --log "$WORK/applications/applications.jsonl" \
    --today "$LATER"
python3 -m autojob next --log "$WORK/applications/applications.jsonl" \
    --today "$LATER" --quiet-days 7
python3 -m autojob next --log "$WORK/applications/applications.jsonl" \
    --today "$LATER" --kind deadline --kind offer
# Batch: apply to a whole folder at once, best fit first. With a fresh
# log and a 0.5 bar, Northwind (0.625) and Harbour (0.562) get packets
# and Vector Labs (0.150) is skipped with the reason. Run it again and
# all three are skipped — the two as already applied, Vector Labs still
# below the bar. With --limit 1 on another fresh log, only Northwind
# lands and Harbour is skipped as beyond the limit. A dry run reports
# the same plan and writes nothing.
python3 -m autojob batch --profile examples/profile.example.json \
    --postings-dir examples --min-score 0.5 \
    --out-dir "$WORK/batch" --today "$(date +%F)"
python3 -m autojob batch --profile examples/profile.example.json \
    --postings-dir examples --min-score 0.5 \
    --out-dir "$WORK/batch" --today "$(date +%F)"
python3 -m autojob batch --profile examples/profile.example.json \
    --postings-dir examples --min-score 0.5 --limit 1 \
    --out-dir "$WORK/batch-limited" --today "$(date +%F)"
python3 -m autojob batch --profile examples/profile.example.json \
    --postings-dir examples --min-score 0.5 \
    --out-dir "$WORK/batch-dry" --dry-run
# Export: the whole log as one row per application — CSV for a
# spreadsheet, JSON for another tool. Eight days on, all three
# applications are in the file with their scores, sources, and latest
# offers; the rejected-only view holds just Harbour and Vector Labs.
python3 -m autojob export --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob export --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --format json --status rejected
python3 -m autojob export --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --out "$WORK/export.csv"
# Backup: the whole search in one zip — the log plus the goals and
# question-bank settings. Eight days on, the backup holds 3
# applications with both sidecars; check it, restore it into a fresh
# directory, and the restored log lists the same three applications.
python3 -m autojob backup --log "$WORK/applications/applications.jsonl" \
    --out "$WORK/search-backup.zip" --today "$FUTURE"
python3 -m autojob backup --check "$WORK/search-backup.zip"
python3 -m autojob restore --in "$WORK/search-backup.zip" \
    --out-dir "$WORK/restored"
python3 -m autojob list --log "$WORK/restored/applications.jsonl"
# Search: find the application by a person, a skill, or a note. Eight
# days on, "Priya" finds the Harbour application through its
# hiring-manager round; "dbt" finds Northwind and Harbour through the
# skills their postings asked for; "salary" finds Harbour through its
# rejection reason. A query whose words live on different applications
# — "Priya PyTorch" — matches nothing, because every word has to
# appear in the one application; and the --status filter narrows the
# "Analytics" search to the rejected stage, which names just Harbour
# (Northwind is still screening).
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "Priya"
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "dbt"
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "salary"
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "Priya PyTorch"
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "Analytics" --status rejected --json
# Resumes: which file each application actually got. Northwind went
# out with the v3 resume (recorded at apply time), Harbour with the
# v2 (recorded while it was live), and Vector Labs has nothing
# recorded. The list shows all three with a roll-up of versions;
# --missing names just Vector Labs; --resume names just Northwind.
# And the v3 file name is searchable like any other recorded fact.
python3 -m autojob resume --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob resume --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --missing
python3 -m autojob resume --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --resume "resume-data-analyst-v3.pdf"
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "v3"
# Blocklist: Vector Labs turned the application down over the ML-stack
# gap, and the profile is not going to grow PyTorch overnight — block
# the company so a good score cannot talk you back into it. The list
# shows the one entry with its reason; rank with --include-applied
# skips it with the reason shown; a batch skips it too; and applying
# to its posting is refused (exit 4) and writes nothing. Removing it
# puts the posting back in play.
python3 -m autojob blocklist --log "$WORK/applications/applications.jsonl" \
    --record --company "Vector Labs" \
    --reason "Rejected over the ML-stack gap; not re-applying." \
    --date "$FUTURE"
python3 -m autojob blocklist --log "$WORK/applications/applications.jsonl"
# Company dossier: everything recorded about one employer on one page.
# The list shows all three companies, Vector Labs flagged blocked; the
# Vector Labs dossier shows its one rejected application and the block
# reason; the Northwind dossier shows its screening application, Sam
# Chen (spoke today), the meetup note, and the offer's first year.
python3 -m autojob company --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob company "Vector Labs" \
    --log "$WORK/applications/applications.jsonl" --today "$FUTURE"
python3 -m autojob company "Northwind Analytics" \
    --log "$WORK/applications/applications.jsonl" --today "$FUTURE"
python3 -m autojob rank --profile examples/profile.example.json \
    --postings-dir examples --log "$WORK/applications/applications.jsonl" \
    --include-applied
python3 -m autojob batch --profile examples/profile.example.json \
    --postings-dir examples --min-score 0.5 \
    --out-dir "$WORK/batch-blocked" \
    --log "$WORK/applications/applications.jsonl" --today "$FUTURE" --dry-run
if python3 -m autojob apply --profile examples/profile.example.json \
    --posting examples/job3.example.md --out-dir "$WORK/applications"; then
    echo "ERROR: apply to a blocked company succeeded" >&2; exit 1
fi
python3 -m autojob blocklist --log "$WORK/applications/applications.jsonl" \
    --remove --company "vector labs"
python3 -m autojob blocklist --log "$WORK/applications/applications.jsonl"
# Audit: one read-only pass over the log — what is missing, stale, or
# inconsistent, with the command that fixes each issue. Sixteen days on,
# the Northwind posting closed 6 days ago and its offer decision was due
# 2 days ago, while the application is still live; the two rejected
# applications have their reasons recorded, so they are clean. At the
# default 14-day bar nothing is stale (the last activity was 8 days
# ago); at a 7-day bar the Northwind application joins the list, and
# --severity error narrows it to just the overdue decision.
python3 -m autojob audit --log "$WORK/applications/applications.jsonl" \
    --today "$LATER"
python3 -m autojob audit --log "$WORK/applications/applications.jsonl" \
    --today "$LATER" --stale-days 7
python3 -m autojob audit --log "$WORK/applications/applications.jsonl" \
    --today "$LATER" --severity error
# Schedule: the week ahead as a calendar, with clashes flagged. Eight
# days on, Northwind has two rounds on the same day — a hiring-manager
# round at 14:30 for 45 minutes and a panel at 15:00 for 30 minutes, so
# the two overlap by 15 minutes — and the Northwind posting closes that
# same day, so the day carries three things. The offer decision falls
# six days out. (Harbour's rounds and offer are not on the schedule:
# that application was rejected, and there is nothing left to schedule
# for it.) The 3-day window shows just the clash day, and the JSON
# carries the same counts.
python3 -m autojob interview --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --date "$INTERVIEW_DAY" --round "Hiring manager" \
    --interviewer "Rana Osei" --mode video --duration-minutes 45 \
    --time 14:30 --today "$FUTURE"
python3 -m autojob interview --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --date "$INTERVIEW_DAY" --round "Panel" \
    --interviewer "Dana Kim" --mode video --duration-minutes 30 \
    --time 15:00 --today "$FUTURE"
python3 -m autojob schedule --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob schedule --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --days 3
python3 -m autojob schedule --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE" --json --out "$WORK/schedule.json"
grep '"conflicts": 1' "$WORK/schedule.json"
# Onboarding: the search ends when an application is hired, but the
# first weeks at the new job still need tracking. Eight days on,
# Northwind hires (screening -> interview -> offer -> hired) and
# three first-week tasks go on the list — the I-9 paperwork due in 2
# days, meeting the onboarding buddy in 5, and the laptop pickup in
# 10. Sixteen days on, the paperwork and the buddy meeting are
# overdue; the paperwork gets done, and `next` names the buddy
# meeting as the overdue onboarding task and the laptop as due soon.
# The timeline shows the tasks being written down and the one being
# finished, and searching for "laptop" finds the hired application
# through its onboarding task.
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --to interview
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --to offer
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --to hired
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --task "Complete I-9 paperwork" \
    --category paperwork --due "$(date -d "$FUTURE +2 days" +%F)" \
    --today "$FUTURE"
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --task "Meet your onboarding buddy" \
    --category people --due "$(date -d "$FUTURE +5 days" +%F)" \
    --today "$FUTURE"
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --record --id APP-001 --task "Pick up laptop" \
    --category equipment --due "$(date -d "$FUTURE +10 days" +%F)" \
    --today "$FUTURE"
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --today "$FUTURE"
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --today "$LATER"
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --done --id APP-001 --number 1 --today "$LATER"
python3 -m autojob onboard --log "$WORK/applications/applications.jsonl" \
    --today "$LATER"
python3 -m autojob next --log "$WORK/applications/applications.jsonl" \
    --today "$LATER" --kind onboarding
python3 -m autojob timeline --log "$WORK/applications/applications.jsonl" \
    --today "$LATER" --kind onboarding
python3 -m autojob search --log "$WORK/applications/applications.jsonl" \
    --query "laptop"
echo "demo OK: packets in $WORK/applications (removed on exit)"
