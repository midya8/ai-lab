# autojob

A command-line helper for job hunting. It drafts applications from a profile you write yourself, and it refuses to invent anything about you. It never submits anything either. Every packet is a draft for you to read and send. Standard library only, works offline.

I built it during my own search because rewriting the same facts for every posting gets old, and the tools that "fill the gaps" do it with plausible fiction.

```bash
cd autojob
bash examples/demo.sh                       # full walkthrough on a fictional profile
python -m autojob validate --profile my-profile.json
python -m autojob match --profile my-profile.json --posting job.md
python -m autojob apply --profile my-profile.json --posting job.md --out-dir applications
python -m autojob next                      # what to do today
python -m unittest discover -s tests        # 890 tests
```

## How it works

- **Profile:** a `profile.json` with your skills (with years), experience bullets, education and preferences. Keep it out of git. Only the fictional example is committed.
- **Posting parser:** pulls out title, company, remote/hybrid, salary range, and required vs preferred skills. Matching is on word boundaries, so `MySQL` doesn't count as `SQL`.
- **Match score:** required coverage (0.6), preferred coverage (0.25) and experience fit (0.15), with every component shown. Missing skills are listed as gaps.
- **Letter:** built only from your own facts. Experience bullets are quoted verbatim, and only matched skills are named. The tests check that a missing skill can never show up in the letter. A `review-notes.md` lists what not to claim.
- **Tracker:** a JSONL log with a fixed state machine (applied, screening, interview, offer, hired, or rejected/withdrawn) and history for every move.

Around that core it grew interview prep, tailored resume sections, follow-up reminders, offer comparison and negotiation briefs, an interview calendar with clash detection, deadlines, weekly goals, rejection reasons, a question bank, notes, contacts, a company page, search, a blocklist, CSV/JSON export, backup/restore, and an audit of the log. The full list is in [COMMANDS.md](COMMANDS.md).

## Demo

On the fictional profile (a data analyst) against a fictional posting, the score is 0.62: 5 of 6 required skills (Communication missing), none of the preferred ones (dbt, Snowflake, A/B testing), and 83% experience fit. The letter quotes three real bullets and names only the five matched skills. The demo then walks two applications through screening, interviews, offers and a rejection, so every command has something to show.

## What it won't do

Invent experience, convert currencies, rewrite your resume, or submit anything. There's no submit command on purpose.
