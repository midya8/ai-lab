# autojob commands

Every command, with the flags I actually use. Run `python -m autojob <command> --help` for the rest.

```bash
bash examples/demo.sh

# check your profile
python -m autojob validate --profile my-profile.json

# score the fit (exit 3 when below --fail-under)
python -m autojob match --profile my-profile.json --posting job.md --fail-under 0.6

# draft a letter (prints to stdout; --out writes a file; --dry-run never writes)
python -m autojob draft --profile my-profile.json --posting job.md --dry-run

# build an application packet and log it
python -m autojob apply --profile my-profile.json --posting job.md \
    --out-dir applications
# ... review applications/<company>-<role>/cover-letter.md, then submit it yourself

python -m autojob list
python -m autojob status --id APP-001 --to interview
python -m autojob report
python -m autojob report --json --stale-days 21 --today 2026-11-30

# build an interview-prep packet (prints Markdown; --out writes a file)
python -m autojob prep --profile my-profile.json --posting job.md

# build a tailored resume section for this posting
python -m autojob tailor --profile my-profile.json --posting job.md

# list applications due a follow-up (drafts included with --profile)
python -m autojob followup --profile my-profile.json

# record a follow-up as sent (resets the follow-up and stale clocks)
python -m autojob followup --record --id APP-001

# one-page digest over the whole log: funnel, stale, due follow-ups,
# and the skills postings keep asking for
python -m autojob digest
python -m autojob digest --json --today 2026-11-30

# gap-closing plan: skills to evidence with a real bullet, skills to
# learn, and the three quick wins
python -m autojob plan --profile my-profile.json
python -m autojob plan --profile my-profile.json --json

# rank a folder of postings by fit: best application first, already-applied
# postings left out unless --include-applied
python -m autojob rank --profile my-profile.json --postings-dir postings/
python -m autojob rank --profile my-profile.json --postings-dir postings/ \
    --include-applied --min-score 0.5 --json

# compare what a folder of postings pays against your salary target
python -m autojob salary --profile my-profile.json --postings-dir postings/
python -m autojob salary --profile my-profile.json --postings-dir postings/ \
    --min-pay 100000 --json

# record an offer on an application, then compare all recorded offers
python -m autojob offer --record --id APP-001 --base 130000 --currency CAD \
    --bonus 10000 --equity 20000 --vesting-years 4 --pto-days 20 \
    --mode remote --deadline 2026-11-25
python -m autojob offer --profile my-profile.json
python -m autojob offer --profile my-profile.json --json

# record an interview round, then see the week ahead
python -m autojob interview --record --id APP-001 --date 2026-11-12 \
    --round "Hiring manager" --interviewer "Priya Nair" --mode video \
    --duration-minutes 45
python -m autojob interview
python -m autojob interview --id APP-001 --json

# list posting closing dates, soonest first — or record one
python -m autojob deadlines
python -m autojob deadlines --overdue
python -m autojob deadlines --record --id APP-001 --date 2026-11-20
python -m autojob apply --profile my-profile.json --posting job.md \
    --deadline 2026-11-20

# weekly application target and streak
python -m autojob goals --set-weekly 2
python -m autojob goals
python -m autojob goals --weeks 4 --json

# record why a rejected application ended, then see the pattern
python -m autojob rejections --record --id APP-001 --reason salary \
    --note "Their band topped out below the target range."
python -m autojob rejections
python -m autojob rejections --reason skills-gap --json

# one prioritised list of what to do: due follow-ups, upcoming
# interviews, closing dates, offer deadlines, quiet contacts,
# stale applications, rejections missing a reason
python -m autojob next
python -m autojob next --kind deadline --kind offer --json

# get the whole log out: one row per application, CSV for a
# spreadsheet or JSON for another tool
python -m autojob export --out applications.csv
python -m autojob export --format json --status rejected

# the questions interviews keep asking, most-asked first — and a
# prepared answer for one of them
python -m autojob questions
python -m autojob questions --record --question "Walk me through your most recent project." --answer "Most recent role first."
python -m autojob questions --unanswered

# find every application that mentions a person, a skill, or a note
python -m autojob search --query "Priya Nair"
python -m autojob search --query "dbt" --status screening --json

# block a company you will not apply to again — apply refuses it,
# rank and batch skip it; remove takes it back off the list
python -m autojob blocklist --record --company "Vector Labs" \
    --reason "Rejected over the ML-stack gap." --date 2026-11-08
python -m autojob blocklist
python -m autojob blocklist --remove --company "Vector Labs"

# which resume file each application was sent with — record it at
# apply time, or afterwards; --missing shows the ones with nothing
# recorded
python -m autojob apply --profile my-profile.json --posting job.md \
    --resume resume-data-analyst-v3.pdf
python -m autojob resume --record --id APP-001 \
    --file resume-data-analyst-v3.pdf --note "shorter summary"
python -m autojob resume
python -m autojob resume --missing

# write down what you learn about an application — a detail from a
# call, a reminder before the next round; the date is required and
# never defaulted
python -m autojob notes --record --id APP-001 --date 2026-11-08 \
    --text "Met the recruiter at the data meetup." --tag company
python -m autojob notes
python -m autojob notes --tag interview

# everything about one company, on one page — or every company in
# the log, most applications first, blocklisted ones flagged
python -m autojob company
python -m autojob company "Northwind Analytics"

# what the interviews actually tested, against the postings — which
# skills came up in the debrief questions, and whether prep called them
python -m autojob review --profile my-profile.json --postings-dir postings/
python -m autojob company "Northwind Analytics"
python -m autojob company

# one read-only pass over the log — what is missing, stale, or
# inconsistent, with the command that fixes each issue
python -m autojob audit
python -m autojob audit --severity error

# the week ahead as a calendar — interviews, offer deadlines and
# closing dates, with clashes flagged
python -m autojob schedule
python -m autojob schedule --days 7 --kind interview

# the brief for one recorded offer — target, other offers, posting
# range, strengths to cite, gaps, and what else is still in play
python -m autojob negotiate --id APP-001 --profile my-profile.json \
    --postings-dir postings/
```

## Exit codes

0 ok · 1 usage error · 2 profile/posting load error · 3 match below
`--fail-under` · 4 draft/apply refused below `--min-coverage` ·
5 duplicate application, or batch stopped on a tracker write ·
6 illegal status transition, follow-up refused,
offer refused, interview refused, contact refused, deadline refused,
rejection refused, source refused, debrief refused, resume refused,
onboard refused, or restore refused (target files already exist).
