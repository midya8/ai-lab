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
    --posting examples/job.example.md --out-dir "$WORK/applications"
python3 -m autojob apply --profile examples/profile.example.json \
    --posting examples/job2.example.md --out-dir "$WORK/applications"
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
echo "demo OK: packets in $WORK/applications (removed on exit)"
