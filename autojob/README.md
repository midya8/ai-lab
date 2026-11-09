# autojob

A job-application assistant that prepares applications from your own
profile — and refuses to invent anything about you. Pure standard-library
Python, 49 tests, no API keys, fully offline.

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
   they are the skills the letter must not claim.
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
   withdrawn as terminal states. Illegal transitions are refused.

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
```

## Demo numbers

On the fictional example (Jordan Lee, Data Analyst, vs a fictional
Northwind Analytics posting): score **0.62 (strong)** — required coverage
83% (5/6, `Communication` missing), preferred coverage 0% (`dbt`,
`Snowflake`, `A/B Testing` missing), experience fit 83%. The draft letter
quotes three verbatim bullets and names only the five matched skills.

## Exit codes

0 ok · 1 usage error · 2 profile/posting load error · 3 match below
`--fail-under` · 4 draft/apply refused below `--min-coverage` ·
5 duplicate application · 6 illegal status transition.

## What it will not do

- Invent experience, skills, employers, or results. If a fact is not in
  your profile, autojob reports it as a gap.
- Edit or rewrite your resume. Attach the real one yourself.
- Submit an application anywhere. There is no submit command on purpose.
- Store your profile in this repo. `profile.json` is yours; the committed
  example is fictional.
