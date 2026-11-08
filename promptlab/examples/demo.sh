#!/usr/bin/env bash
# End-to-end demo: version three prompts, A/B them, then gate on a baseline.
set -e
cd "$(dirname "$0")/.."
ROOT=$(mktemp -d)
P="python -m promptlab --root $ROOT"
CASES=examples/ticket_cases.jsonl
$P commit triage --file examples/triage_v1.txt -m "naive baseline"
$P commit triage --file examples/triage_v2.txt -m "add categories, one-word answer"
$P commit triage --file examples/triage_v3.txt -m "delimiters, example, JSON"
$P tag triage prod --ref v1
$P log triage
$P diff triage v1 v2
echo; $P compare triage v1 v2 --cases $CASES
echo; $P compare triage v1 v3 --cases $CASES
$P baseline triage --ref v2 --cases $CASES --out "$ROOT/baseline.json"
$P check triage --ref v3 --cases $CASES --baseline "$ROOT/baseline.json" || true
echo; python -m promptlab calibrate --labels examples/judge_labels.jsonl --judge rule
