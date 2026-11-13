# autojob

A job-application assistant that prepares applications from your own
profile — and refuses to invent anything about you. Pure standard-library
Python, 439 tests, no API keys, fully offline.

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
    mode (phone / video / onsite), duration, an optional start time
    (`--time 14:30`, 24-hour; a round without one is an all-day event
    when exported to a calendar), and a note. Validation
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
16. **Calendar** (`autojob/calendar.py`, v0.12) — `autojob calendar
    --out search.ics` exports the next 14 days as an .ics file any
    calendar app can import: interview rounds in the window, offer
    decision deadlines in the window, posting closing dates in the
    window (v0.15), and follow-ups due as of today
    (dated today, because the action is due now — the description says
    so). A round with a start time exports as a timed event, 45 minutes
    from 14:30 ending at 15:15; a round with a time but no recorded
    duration is shown as one hour and the description says so; a round
    with no time is an all-day event. Times are wall-clock times exactly
    as recorded, with no timezone stored and none invented, so the file
    uses floating local times. Deadlines and interviews on terminal
    applications (hired, rejected, withdrawn) are left out. UIDs are a
    hash of kind, application, date, and summary, so exporting twice
    produces the same file and importing twice does not duplicate
    events. `--days N` sets the window (0 means today only), `--wait-days`
    overrides the follow-up wait, `--json` prints the events instead of
    .ics, `--today` pins the reference date.
17. **Contacts** (`autojob/contacts.py`, v0.13) — `contacts --record
    --id APP-001 --name "Sam Chen" --role Recruiter --channel linkedin
    --date 2026-11-01` stores a person on a tracker entry: name, role,
    channel (email / linkedin / phone / in-person / other), the date you
    first reached them or last spoke to them, and a note. Validation
    reports every problem at once, and the date is required — it is
    never defaulted to today, because writing down a name you met last
    week should not silently claim you spoke to them today. There is
    deliberately no email or phone field: a name plus a channel is
    enough to know who someone is and how you reached them. Recording
    the same person again is a new touch, not a duplicate, so the list
    shows one row per person with the latest touch, how many touches
    there have been, and when you first reached them. `contacts` on its
    own lists everyone on a live application, quietest first: a contact
    is quiet when you have not spoken to *them* for longer than
    `--quiet-days` (default 14; exactly N days is not quiet yet), and an
    interview with someone else or an offer arriving does not count as
    speaking to this person. Recording a contact is refused on a
    terminal application and does not move the application's status, the
    same rule as offers and interviews; it is stored as a history event
    too, so a contact counts as activity for follow-ups and the stale
    check, and it never splits a stage stay in the analytics. `--id
    APP-001` shows one application's contacts only, `--json` prints the
    list as JSON, `--out` writes it, `--today` pins the reference date.
18. **Timeline** (`autojob/timeline.py`, v0.14) — `timeline` lays out
    everything the other commands already recorded as one feed across
    all applications, newest first: applications (with the match score),
    status moves, follow-ups, interviews, contacts, and offers, grouped
    by day. An interview line names the round, the interviewer, the
    mode, the duration, and the start time exactly as recorded; an
    offer line states the base, bonus, and equity exactly as recorded.
    The details are paired to their records, not assumed: an interview
    event carries its date and round label, a contact event its date
    and name, and the n-th offer event on an application is the n-th
    offer in its list — the two are appended in the same order. When a
    record is missing, the line falls back to the bare fact ("Offer
    recorded") instead of guessing figures, and an event with a bad
    date or an unknown kind is skipped, never fatal, so one hand-edited
    log line cannot hide the rest of the feed. `--since` / `--until`
    bound the feed by date (inclusive on both ends), `--kind` filters
    to one kind of event and repeats, `--id` shows one application,
    `--oldest` flips the feed to chronological, `--json` prints it as
    JSON, `--out` writes it, `--today` pins the reference date.
19. **Closing dates** (`autojob/deadlines.py`, v0.15) — `deadlines` lists
    the closing date of every live application, soonest first, so the
    posting that stops taking applications tomorrow is the one you
    finish tonight. A closing date is recorded when you apply (`apply
    --deadline YYYY-MM-DD`) or later (`deadlines --record --id APP-001
    --date YYYY-MM-DD`); re-recording moves it, because a posting's
    closing date can change. Each row shows the days left and its
    state — overdue (the date passed while the application is still
    live, so the posting may have closed under you), today, closing
    soon (1–7 days out; exactly 7 is soon, not later), or later. A
    past date is allowed when you record it — a posting can already
    be closed, and the list flags it overdue instead of refusing a
    fact. `--days N` narrows the list to deadlines N days out or
    sooner (overdue ones are always shown), `--overdue` keeps only
    the overdue rows, and the header counts the live applications
    with no closing date recorded instead of inventing one. A closing
    date is a fact about the posting, not something that happened to
    the application, so it is stored as a field on the tracker entry,
    not a history event: recording one does not reset the follow-up
    or stale clocks, and it does not appear in the timeline feed.
    Recording a closing date is refused on a terminal application
    (hired, rejected, withdrawn) and on an unknown id. Posting
    closing dates in the calendar window also export as all-day
    events from `calendar`. `--json` prints the list as JSON, `--out`
    writes it, `--today` pins the reference date.
20. **Weekly goals** (`autojob/goals.py`, v0.16) — `goals --set-weekly 2`
    sets a target of two applications a week, kept in a small
    `goals.json` beside the log; the log itself is never touched, so
    setting a target is a setting, not an event, and it shows up in
    no history and resets no clock. `goals` on its own shows this
    week against the target (a week is Monday to Sunday, and an
    application counts in the week of its `applied_at` date), the
    current streak of weeks that met the target, the best streak,
    and a table of the last 8 weeks (`--weeks N` changes how far
    back it runs). Exactly the target counts as met. The streak
    walks back from the current week when it already met the target,
    otherwise from the last completed week — the current week is
    never judged a miss while it is still in progress, and weeks
    before your first logged application carry no verdict at all
    instead of being scored as misses for a goal that did not exist
    yet. With no target set there is nothing to meet and no streak
    to count, so the command shows the weekly counts only and says
    so; `--set-weekly 0` clears the target. An entry whose
    `applied_at` will not parse is skipped and counted, never fatal.
    `--json` prints the view as JSON, `--out` writes it, `--today`
    pins the reference date, `--goals` points at a different goals
    file.
21. **Rejection analysis** (`autojob/rejections.py`, v0.17) — `rejections`
    lists every rejected application with the stage it reached, how
    long it took to hear back, and the reason you recorded, most
    recently rejected first. `rejections --record --id APP-001 --reason
    salary --note "Their band topped out below the target range."`
    stores the reason on an application that is already `rejected` —
    a live application has not been rejected yet, and a hired or
    withdrawn one was not rejected at all, so recording is refused
    for both. The reason is one of nine fixed categories (`ghosted`,
    `skills-gap`, `experience-level`, `salary`, `location`,
    `internal-hire`, `role-closed`, `culture-fit`, `other`); the note
    is what you were actually told, in your own words. The stage
    reached comes from the tracker's history — the status immediately
    before the move to `rejected` — and the days to a decision run
    from `applied_at` to that move; a row whose dates will not parse
    shows no day count instead of a guessed one. The summary counts
    rejections by reason (including how many carry no reason yet) and
    by furthest stage reached, gives the median days to a decision,
    and rolls up the skills those postings asked for that the profile
    lacked — a pattern to look at, not a cause, since a missing skill
    on a rejected application does not prove it is why you were
    turned down. Recording a reason is a field on the entry, not a
    history event: it does not reset the follow-up or stale clocks
    and does not appear in the timeline feed, and re-recording
    replaces the reason when you learn the real one later. `--reason`
    filters the list to one reason, `--json` prints it as JSON,
    `--out` writes it, `--today` pins the reference date.

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
The demo closes with a calendar export 8 days on: 4 events in the next
14 days — the hiring-manager round at 14:30 for 45 minutes, the Northwind
posting closing date 2 days out, and both offer deadlines 6 days out;
no follow-up is due, because both applications had activity recorded that
day. The demo closes with two contacts 8 days on: Sam Chen, the Northwind recruiter (LinkedIn, spoke
today), and Alex Rivera, the Harbour hiring manager (email, 8 days
since last spoke — quiet at `--quiet-days 7`, not at the default 14).
The demo closes with two closing dates 8 days on: the Northwind posting
closes in 2 days and the Harbour posting in 22 days, nothing overdue;
the Harbour listing is then updated to close in 17 days.
The demo closes with a timeline 8 days on: 12 events, newest first,
from the day both applications were filed to the hiring-manager round
two days out — 2 applied, 3 status moves, 1 follow-up, 2 offers,
2 interviews, 2 contacts; the filtered `--kind interview --kind offer
--oldest` view shows just those 4 events, chronologically.
The demo closes with a weekly goal of 2 applications: both were filed
this week, so the target is met and the streak is 1 week; 8 days on,
the new week shows 0 of 2 with 2 to go, and the streak still stands
at 1, because last week met the target — the weeks before the first
application carry no verdict at all.
The demo closes with two rejections: the Harbour application, which
reached the interview stage, is recorded as ending over salary, and
the Vector Labs application, which never got past the application,
as a skills gap (the profile has neither PyTorch nor TensorFlow).
Both were applied to and rejected on the same day, so the report
shows a median of 0.0 days to a decision; the missing-skills roll-up
names Communication (required, on the Harbour rejection) and the ML
stack — PyTorch, TensorFlow, Statistics, Machine Learning — on the
Vector Labs one; the `--reason skills-gap` filter shows just the
Vector Labs row.

## Exit codes

0 ok · 1 usage error · 2 profile/posting load error · 3 match below
`--fail-under` · 4 draft/apply refused below `--min-coverage` ·
5 duplicate application · 6 illegal status transition, follow-up refused,
offer refused, interview refused, contact refused, deadline refused,
or rejection refused.

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
