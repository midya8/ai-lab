# autojob

A job-application assistant that prepares applications from your own
profile — and refuses to invent anything about you. Pure standard-library
Python, 107 tests, no API keys, fully offline.

Built in November 2026 as a portfolio project. It is not client work, it is
not affiliated with any employer, and the example profile and posting in
`examples/` are fictional demo data, not a real person or a real job. The
match numbers below are rates on that small demo posting, not a benchmark
claim.

## The problem it solves

Applying for jobs means rewriting the same facts into a new shape for every
posting, and the temptation is to let a tool fill the gaps with plausible
inventions. autojob does the boring half — parsing the posting, scoring the
fit, drafting a letter from facts you actually supplied — and makes the
gaps visible instead of filling them. It never submits anything: the packet
it builds is a draft for a human to review and send.

## How it works

1. **Profile** (`autojob/profile.py`) — a `profile.json` you write yourself:
   name, headline, contact, skills with years, experience with verbatim
   bullets, education, and preferences (remote, notice period, salary
   range). Strict validation reports every problem at once; a bad date or
   an inverted salary range fails loudly. Your profile file is your data —
   keep it out of git; only the fictional `examples/profile.example.json`
   is committed.
2. **Posting parser** (`autojob/posting.py`) — extracts title, company,
   location, remote/hybrid/onsite signals, salary range, and required vs
   preferred skills from a posting's Markdown or plain text. Skills are
   found by word-boundary keyword matching against a vocabulary built
   from your profile's skills plus a common data/BI/engineering list —
   `MySQL` does not count as `SQL`, and a lone `R` never matches
   "Reporting".
3. **Matching** (`autojob/match.py`) — a transparent score, every component
   reported: required coverage (0.6), preferred coverage (0.25),
   experience fit (0.15, the share of required skills backed by at least
   one year in your profile). A component with nothing to measure is left
   out of the average, never scored as 1.0. Gaps are listed explicitly:
   they are the skills the letter must not claim. `apply` also writes an
   `interview-prep.md` into every packet, built by the prep module below.
4. **Letter** (`autojob/letter.py`) — the draft is assembled only from
   profile facts: your headline and summary, experience bullets quoted
   verbatim (with role and company), and skills the profile actually has
   that the posting asks for. A missing skill cannot appear in the letter
   — the code only ever iterates over matched skills, and the tests assert
   it. A companion `review-notes.md` lists the gaps ("do not claim"),
   salary mismatches, and a pre-submit checklist.
5. **Tracker** (`autojob/tracker.py`) — a JSONL log of applications with
   dedupe on company + title (applying twice is refused; use `status` to
   update the existing entry) and a fixed state machine:
   applied → screening → interview → offer → hired, with rejected /
   withdrawn as terminal states. Illegal transitions are refused. Every
   status move is recorded in the entry's `history` (status + date), so
   the log knows when each stage was entered. Entries logged before
   v0.2 have no history; their first recorded move backfills the stage
   they were in at `applied_at`, so durations for those entries are
   upper bounds, not measured stays.
6. **Analytics** (`autojob/analytics.py`, v0.2) — `report` turns the log
   into a funnel: how many applications reached each stage and the
   conversion between stages, response and rejection rates, median days
   spent in each stage (completed stays only — the current stay has no
   end date yet), match-score averages split by outcome (reached
   interview+ vs rejected at applied), applications with no movement in
   `--stale-days` (default 14), and a per-company roll-up. Everything is
   arithmetic on your own log; the report says `n=` wherever a number
   comes from a small sample. `--json` prints the same numbers as JSON;
   `--today YYYY-MM-DD` pins the reference date for stale checks.
7. **Interview prep** (`autojob/prep.py`, v0.3) — `prep` builds a study
   sheet for the interview from the same facts: talking points pairing
   each matched skill with your years/level and the verbatim bullets
   that mention it (a matched skill with no bullet is flagged as
   unevidenced, not dressed up), gaps to prepare an honest answer for
   (never claimed), likely questions filled only with posting/profile
   facts, your negotiation facts (notice period, salary range, remote
   preference), and a story bank quoting your experience verbatim, most
   recent first. `--json` prints the packet as JSON; `--out` writes it.
8. **Follow-ups** (`autojob/followup.py`, v0.4) — `followup` lists the
   applications that have gone quiet past their stage's wait: 7 days at
   applied/screening, 3 days after an interview, 2 days at offer stage
   (`--wait-days` overrides them all; exactly N days is not due yet, the
   same boundary the stale check uses). Each due item comes with a short
   draft email built only from tracker facts — role, company, date
   applied — plus your name from `--profile`; the greeting is "Hiring
   team" because the tracker knows no recruiter name, and none is
   invented. `followup --record --id APP-001` records a follow-up as
   sent: it is stored as a history event, which resets both the
   follow-up clock and the stale clock, and a second follow-up on the
   same day (or one on a terminal application) is refused. Recording a
   follow-up mid-stage does not split the stage stay in the analytics —
   a stay is measured between status changes only. `--json` prints the
   due list with drafts; without `--profile` the drafts are omitted
   rather than written to nobody.
9. **Tailored resume** (`autojob/tailor.py`, v0.5) — `tailor` builds the
   facts a resume needs for *this* posting: a skills matrix covering
   every skill the posting asks for (a skill you have shows years/level
   and the verbatim bullets that mention it; a skill you do not have is
   a gap, never a claim), the order to list your skills in (matched
   skills sorted by years, most first, required before preferred),
   experience roles ranked by how many distinct matched posting skills
   their bullets mention (most recent first on ties; a role with no
   evidenced skill ranks last and is flagged as unevidenced for this
   posting), summary ingredients (your headline verbatim plus the top
   lead skills with years — ingredients for a summary you write
   yourself, not a written summary), and your education verbatim.
   `apply` also writes `tailored-resume.md` into every packet. `--json`
   prints the section as JSON; `--out` writes it.

## Quick start

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
```

## Demo numbers

On the fictional example (Jordan Lee, Data Analyst, vs a fictional
Northwind Analytics posting): score **0.62 (strong)** — required coverage
83% (5/6, `Communication` missing), preferred coverage 0% (`dbt`,
`Snowflake`, `A/B Testing` missing), experience fit 83%. The draft letter
quotes three verbatim bullets and names only the five matched skills. The
demo also applies to a second fictional posting (Harbour Analytics, score
0.56) and moves the two applications through screening and interview, so
the closing `report` shows a real two-application funnel: 2 applied,
2 screening, 1 interview. The demo also builds an interview-prep packet
for the Northwind role: 5 talking points (SQL 4 yrs with a verbatim
bullet, Power BI 3 yrs, and `Stakeholder Management` flagged as having no
evidencing bullet), 4 gaps to prepare for honestly, and 8 likely questions.
Eight days on, the demo's `followup` step finds both applications due
(the 7-day applied/screening wait and the 3-day interview wait), records
a follow-up for the Northwind application, and shows the Harbour one
still due. The demo also builds a tailored resume section for the
Northwind role: 5 skills to lead with (SQL 4 yrs first, then Power BI,
Python, and Stakeholder Management at 3 yrs, then Tableau at 2 yrs), a
9-row skills matrix (5 matched, 4 gaps — `Communication`, `dbt`,
`Snowflake`, `A/B Testing` — each flagged "do not claim"), and the
experience ranked with the Data Analyst role first (evidences SQL and
Power BI) ahead of Reporting Analyst (evidences Python only).

## Exit codes

0 ok · 1 usage error · 2 profile/posting load error · 3 match below
`--fail-under` · 4 draft/apply refused below `--min-coverage` ·
5 duplicate application · 6 illegal status transition or follow-up refused.

## What it will not do

- Invent experience, skills, employers, or results. If a fact is not in
  your profile, autojob reports it as a gap.
- Edit or rewrite your resume. Attach the real one yourself.
- Submit an application anywhere. There is no submit command on purpose.
- Store your profile in this repo. `profile.json` is yours; the committed
  example is fictional.
