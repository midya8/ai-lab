# promptlab

Treat prompts like code: version them, A/B them with real statistics, and
block a prompt change in CI when it breaks cases that used to work.
Pure standard-library Python, 29 tests, no API keys needed.

Built in November 2026 as a portfolio project. The bundled `RuleModel` is a
**deterministic stand-in for an LLM** whose error rate depends on prompt
wording (invented rates, not measurements of any real model), so the demo is
reproducible offline. Point `--model cmd:<command>` at a real model to use it
for real.

## Why

"I rewrote the prompt and it feels better" is not evidence. Prompt changes
fix some inputs and silently break others, and on a 20-case test set a
5-point swing is usually noise. promptlab makes the three questions explicit:

1. **Which exact text was run?** Immutable, numbered, content-hashed versions with movable tags.
2. **Is B really better than A?** Paired comparison on the same cases: exact McNemar test on the cases that flipped, Wilson intervals on each pass rate, and a bootstrap CI on the mean score change.
3. **Did this change break anything that worked?** A baseline gate that fails on *any* previously-passing case that now fails, even when the average goes up.

## Quick start

```bash
bash examples/demo.sh          # end-to-end run in a temp registry
python -m unittest discover -s tests
```

```bash
python -m promptlab --root prompts commit triage --file triage.txt -m "add categories"
python -m promptlab --root prompts tag triage prod --ref v2
python -m promptlab --root prompts diff triage v1 v2
python -m promptlab --root prompts run triage --ref prod --cases cases.jsonl -v
python -m promptlab --root prompts compare triage v1 v3 --cases cases.jsonl --report ab.md
python -m promptlab --root prompts baseline triage --ref prod --cases cases.jsonl --out baseline.json
python -m promptlab --root prompts check triage --ref latest --cases cases.jsonl --baseline baseline.json   # exit 1 on regression
```

Refs accepted everywhere: `latest`, `v3` / `3`, a tag, or a hash prefix (4+ chars).

## Example result (synthetic, from `examples/demo.sh`)

Three versions of a ticket-triage prompt on 60 generated tickets:

| Version | Change | Pass rate | 95% CI |
|---|---|---|---|
| v1 | `Classify the support ticket.` | 35/60 = 58.3% | 45.7–69.9% |
| v2 | + category list, one-word answer | 54/60 = 90.0% | 79.9–95.3% |
| v3 | + `<ticket>` delimiters, worked example, JSON output | 59/60 = 98.3% | 91.1–99.7% |

v1 vs v2: *B is better (exact McNemar p = 0.0005)* — 24 cases fixed, 5 broken.
Gating v3 against a v2 baseline then **fails** despite the higher average:
`1 regression(s) > allowed 0: t036`. That is the point of the gate.

## Non-deterministic models

`--samples N` calls the model N times per case; `--threshold F` (default 1.0) is the fraction of samples that must pass every check. Mixed outcomes are flagged as flaky and listed after `run -v`; case score is the mean over samples.

```
promptlab run summarize --cases cases.jsonl --model cmd:./my_llm.sh --samples 5 --threshold 0.8 -v
```

## Test cases

JSONL, one case per line (`#` comments allowed):

```json
{"id": "t001", "vars": {"ticket": "I was charged twice"}, "checks": [{"type": "regex", "value": "\\bbilling\\b", "ignore_case": true}]}
```

Checks: `contains`, `not_contains`, `equals`, `regex`, `max_words`, `min_words`,
`max_chars`, `json_valid`, `json_field` (dotted path, optional `value`).
A case passes only if all its checks pass; its score is the fraction passed.
A bad check, a render error or a failing model call fails that case and is
recorded — it never aborts the run.

## Templates

`{{name}}` and `{{name|default}}`. Single braces (JSON examples inside the
prompt) are left alone. Rendering is strict: a missing required variable or a
misspelled extra one is an error, because a typo'd variable would otherwise
silently test the wrong prompt.

## Design notes

- **Paired, not independent.** Both prompts see the same cases, so only the discordant cases carry information; McNemar on those has far more power than comparing two independent proportions.
- **Underpowered is said out loud.** With few discordant cases the verdict is "no significant difference" plus a note to add cases, never "equal".
- **Unchanged commits are no-ops** (whitespace-insensitive hash), so re-running a deploy script creates no phantom versions.
- **Registry is plain JSON files** under `<root>/<name>/vNNNN.json`, so it diffs and reviews in git.
- **`cmd:` adapter** runs any executable (prompt on stdin, answer on stdout) without a shell.

## Layout

| File | Role |
|---|---|
| `template.py` | placeholders, strict rendering, content digest |
| `registry.py` | versions, tags, ref resolution, diff |
| `checks.py` | deterministic output assertions |
| `runner.py` | case loading, running, per-case results, save/load |
| `stats.py` | Wilson, exact McNemar, paired bootstrap |
| `compare.py` | A/B verdict, Markdown report, baseline regression gate |
| `models.py` | `RuleModel` offline stand-in, `CommandModel` adapter |

## Limitations

Deterministic checks only (no LLM-as-judge here; see `llm-eval-harness` and
`evalkit`). Real models are non-deterministic, so for those, run each case
several times or fix temperature to 0 before trusting a small difference.
