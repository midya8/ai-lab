# autojob

A job-application assistant that prepares applications from your own
profile — and refuses to invent anything about you. Pure standard-library
Python, 254 tests, no API keys, fully offline.

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
10. **Digest** (`autojob/digest.py`, v0.6) — `digest` is one page over the
    whole log: the funnel, median days in stage, match-score summary,
    stale applications, applications due a follow-up, and a gap
    leaderboard — the skills postings keep asking for that the profile
    does not have, counted across every application that recorded them
    (most-missed first, required and preferred counted separately, with
    the companies that asked for each). `apply` records the posting's
    required/preferred skills and which of them the profile matched in
    the tracker entry, so the count is arithmetic on the log, not a
    guess. Entries logged before v0.6 carry no skill lists; the digest
    says how many it skipped instead of counting them as if they had
    none. `--json` prints the same numbers as JSON; `--out` writes the
    digest; `--today` pins the reference date, `--stale-days` and
    `--wait-days` override the stale and follow-up windows.
11. **Plan** (`autojob/plan.py`, v0.7) — `plan` turns the digest's gap
    leaderboard into work to do, and adds the other half of the gap
    story: the skills the profile has that postings asked for but no
    experience bullet evidences. For those it names the skill, its
    years/level, which postings asked for it, and the most recent role
    as the suggested place to add a real bullet — a suggestion, not a
    claim, and the plan never writes a bullet for you. For skills the
    profile does not have at all it reuses the digest leaderboard
    (most-missed first, required and preferred counted separately) and
    says plainly: do not claim it until you have it. `quick_wins`
    lists the three fixes that help the most logged applications —
    an evidence item helps every application that asked for the skill,
    a learn item helps every application where it was missing, and a
    one-bullet fix sorts before learning a skill on a tie. Entries
    logged before v0.6 carry no skill lists and are counted as skipped,
    not guessed. `--json` prints the plan as JSON; `--out` writes it;
    `--today` pins the reference date.
12. **Rank** (`autojob/rank.py`, v0.8) — `rank` scores every posting in a
    folder against the profile with the same transparent match as
    `match`, and lists them best-first so the strongest application
    comes first: score, then required coverage, then title, so ties
    break the same way every run. Postings already in the application
    log are left out by default — ranking is about where to apply
    *next* — and are flagged with their application id when
    `--include-applied` is given; the "best next application" line
    always names the highest-ranked posting not yet applied. A posting
    with no recognisable skills scores `None` and sorts last, labelled
    unmeasurable rather than given a made-up zero; a file that cannot
    be read (not UTF-8) is reported in `skipped` and never crashes the
    run. `--min-score` drops postings below a threshold, `--json`
    prints the ranking as JSON, `--out` writes it. `--log` points at
    the application log used for the already-applied check.
13. **Salary** (`autojob/salary.py`, v0.9) — `salary` compares what a
    folder of postings pays against the salary target in the profile's
    preferences, best-paying first: each row carries the range as
    listed, its midpoint, and a verdict — `meets target`,
    `below target` (its max is under your minimum), `above target`
    (its min is over your maximum), `no range listed`, or
    `not compared` with the reason. Currencies are never converted:
    a range is compared only when it states the same three-letter
    currency code as the target, so a bare `$` or a USD range against
    a CAD target is reported, not guessed. A posting that does not
    state a period is treated as annual — the assumption is printed
    on every report — and an hourly rate is listed but never compared
    with an annual target. Medians are per currency, over ranged,
    non-hourly postings. Already-applied postings are flagged with
    their application id. `--min-pay` drops postings whose max is
    below a floor, but only rows in the target's currency, which is
    the only currency a floor in the target's terms can judge; rows
    that cannot be judged are kept and labelled. `--json` prints the
    analysis as JSON, `--out` writes it. `parse_salary` now also
    captures a currency code written after the figures
    (`$95,000 - $120,000 CAD` → `CAD`).
14. **Offers** (`autojob/offer.py`, v0.10) — `offer --record --id APP-001
    --base 130000 --currency CAD` stores an offer on a tracker entry:
    base, annual bonus, equity with an optional vesting schedule, paid
    time off, work mode, start date, decision deadline, and a note.
    Validation reports every problem at once; a base stated hourly is
    annualised at 2,080 hours a year, monthly at 12 months, and the
    stated figure is kept next to the annualised one. Recording an
    offer is refused on a terminal application (hired, rejected,
    withdrawn) and on an unknown id, and it does not move the
    application's status — that stays explicit through `status`. `offer`
    on its own compares every recorded offer against the salary target
    in the profile's preferences: one row per application, using its
    latest offer, ranked by first-year total (base + bonus + equity per
    vesting year) among offers in the target's currency, with a verdict
    on the annual base — `meets target`, `below target`, `above target`,
    or `not compared` with the reason. Currencies are never converted,
    so an offer in another currency is listed unranked, not guessed;
    with no target at all the offers are listed, not ranked. Equity
    counts only when a vesting schedule is given; equity with no
    vesting schedule is listed, never counted. A bonus is assumed
    annual and the report says so. Each row shows days to the decision
    deadline and flags a deadline already passed. `--json` prints the
    comparison as JSON, `--out` writes it, `--today` pins the
    reference date for deadlines.
15. **Interviews** (`autojob/interview.py`, v0.11) — `interview --record
    --id APP-001 --date 2026-11-12 --round "Hiring manager"
    --interviewer "Priya Nair" --mode video --duration-minutes 45`
    stores a round on a tracker entry: date, round label, interviewer,
    mode (phone / video / onsite), duration, and a note. Validation
    reports every problem at once. Recording a round is refused on a
    terminal application (hired, rejected, withdrawn) and on an unknown
    id, and it does not move the application's status — that stays
    explicit through `status`, the same rule as offers. Several rounds
    per application are allowed, including two on the same day, so a
    round is never deduplicated. `interview` on its own lists the week
    ahead: upcoming rounds soonest first (today counts as upcoming,
    with 0 days to go) and past rounds most recent first, each with
    its days to go or days since. `--id APP-001` shows one
    application's rounds only. `--json` prints the schedule as JSON,
    `--out` writes it, `--today` pins the reference date. Everything
    recorded is a fact the user supplies — the round label, the
    interviewer, and the notes are the user's own words; this module
    never generates text about an interview, so it cannot invent an
    interviewer, a question, or an outcome.

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
Power BI) ahead of Reporting Analyst (evidences Python only). The
demo closes with a digest 8 days on: funnel 2 applied, 2 screening,
1 interview; 1 application due a follow-up (Harbour, 8 days waiting);
and a gap leaderboard led by `Communication` (missing as a required
skill in 2 of 2 applications), then `dbt` and `Snowflake` (missing as
preferred skills in 2 of 2), then `A/B Testing` and `Looker` (1 of 1
each). The demo closes with a gap-closing plan for the same two
applications: quick wins are learning `Communication`, `dbt`, and
`Snowflake` (each missing in 2 of 2); two skills the profile has but no
bullet proves — `Stakeholder Management` (3 yrs) and `Tableau` (2 yrs),
both asked by Northwind — with the most recent role (Data Analyst at
Northwind Traders) suggested as the place to add a real evidencing
bullet; and the same five skills to learn, led by `Communication`.
The demo closes with a ranking of the three example postings: Northwind
first (0.625), Harbour second (0.562), and a third fictional posting —
Machine Learning Engineer at Vector Labs (Python matched, but `TensorFlow`,
`PyTorch`, `Machine Learning`, and `Statistics` all missing) — last at
0.150. The plain rank excludes the two already-applied postings and names
Vector Labs as the only one left to apply to; `--include-applied` shows
all three, flagged with their application ids. The demo closes with a
salary analysis of the same three postings against the profile's CAD
95,000–120,000 target: Vector Labs pays above it (130,000–160,000),
Northwind and Harbour meet it, and the CAD medians are 105,000–130,000
(median midpoint 117,500); the two applied postings are flagged with
their application ids. The demo closes with an offer comparison: the
Northwind offer (CAD 130,000 base + 10,000 bonus + 20,000 equity over
4 years) is CAD 145,000 in the first year and above target on base
alone, the Harbour offer (CAD 105,000 + 5,000 bonus) is CAD 110,000
and meets the target, Northwind ranks first, and both show 6 days left
to decide on their 14-day deadlines. The demo closes with an interview
schedule for the Harbour application, 8 days on: a recruiter screen
the day before (phone, 30 minutes) and a hiring-manager round two days
ahead (video, 45 minutes, with Priya Nair) — 1 upcoming, 1 past.

## Exit codes

0 ok · 1 usage error · 2 profile/posting load error · 3 match below
`--fail-under` · 4 draft/apply refused below `--min-coverage` ·
5 duplicate application · 6 illegal status transition, follow-up refused,
offer refused, or interview refused.

## What it will not do

- Invent experience, skills, employers, or results. If a fact is not in
  your profile, autojob reports it as a gap.
- Convert currencies, or compare an hourly rate with an annual target.
  A range is compared only when it states the same currency code as
  your target; anything else is reported, not guessed.
- Edit or rewrite your resume. Attach the real one yourself.
- Submit an application anywhere. There is no submit command on purpose.
- Store your profile in this repo. `profile.json` is yours; the committed
  example is fictional.
