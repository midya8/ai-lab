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
python3 -m autojob list --log "$WORK/applications/applications.jsonl"
python3 -m autojob status --log "$WORK/applications/applications.jsonl" \
    --id APP-001 --to screening
python3 -m autojob report --log "$WORK/applications/applications.jsonl"
echo "demo OK: packet in $WORK/applications (removed on exit)"
