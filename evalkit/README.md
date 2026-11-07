# evalkit

A small, dependency-free LLM evaluation harness for catching prompt and model
regressions before they ship.

Built in November 2026 as a portfolio project. It is not client work, it is
not affiliated with any employer, and it makes no benchmark claims — it is a
tool for running your own golden sets against your own models and noticing
when something that used to pass starts failing.

## The problem it solves

Prompt edits and model upgrades break things silently. A summariser that
used to mention the key number stops mentioning it; a JSON extractor starts
returning malformed output; a refusal that used to hold starts complying.
evalkit gives you a repeatable check for that: a versioned set of prompts
with deterministic pass/fail rules, run on every change, diffed against the
last known-good run.

## How it works

1. **Golden sets** — versioned JSON files under `cases/`. A case is a prompt
   plus the checks its response must satisfy: strings that must appear,
   strings that must not appear, regex matches, JSON validity, length
   bounds, and rubric items (each a contains/regex rule with a name).
   Cases are validated before anything runs: duplicate ids, empty prompts,
   invalid regexes, and cases with no checks at all are rejected.
2. **Scorers** — every check is a plain, inspectable rule. A case's score
   is the fraction of its checks that pass; 1.0 means everything passed.
   No LLM-as-judge by default — judging with a model needs a key and makes
   runs non-deterministic, which defeats the point of a regression harness.
3. **Adapters** — a pluggable `generate(prompt, seed) -> str` interface.
   Ships with a deterministic stub adapter (canned responses from a
   fixtures file, for tests and demos) and an OpenAI-compatible HTTP
   adapter over plain `urllib` (key from the environment only).
4. **Regression detection** — `evalkit compare` diffs two run files and
   lists newly failing cases, newly passing cases, and score deltas per
   tag. It exits 1 if anything regressed or the overall score drops below
   `--min-score`, so it works as a CI gate.

## Quickstart

Requires Python 3.10+ and nothing else (stdlib only, works fully offline
with the stub adapter).

```bash
# Validate the demo golden set
python -m evalkit validate cases/demo.json

# Run it against the stub adapter (repeat twice: checks determinism)
python -m evalkit run cases/demo.json --adapter stub \
    --fixtures fixtures/stub_responses.json \
    --out runs/run.json --repeat 2 --seed 42

# Render reports
python -m evalkit report runs/run.json --md reports/run.md --json reports/run.json

# Diff against a known-good baseline; exits 1 on regression
python -m evalkit compare runs/baseline.json runs/run.json --min-score 0.85
```

Run only a slice of the set:

```bash
python -m evalkit run cases/demo.json --adapter stub \
    --fixtures fixtures/stub_responses.json --only json
python -m evalkit run cases/demo.json --adapter stub \
    --fixtures fixtures/stub_responses.json --filter extract
```

Use a real model (key from the environment only, never from a file):

```bash
export OPENAI_API_KEY="..."
python -m evalkit run cases/demo.json --adapter openai --model gpt-4o-mini \
    --out runs/run.json
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | All cases passed / no regression |
| 1 | Failures, flaky cases, or a regression |
| 2 | Invalid golden set, bad adapter, or unusable run file |

## Example: catching a regression

The demo set has 14 cases. `runs/baseline.json` is the same stub with the
two deliberate failures fixed, so it scores 1.000. The demo run scores
0.905 and `compare` flags exactly what broke:

```
baseline overall score: 1.000
run      overall score: 0.905 (delta -0.095)
min-score threshold:    0.850

newly failing (2):
  - extract-002
  - refuse-002
newly passing: none

score delta by tag:
  classify: +0.000
  code: +0.000
  email: +0.000
  extract: -0.167
  format: +0.000
  json: -0.111
  qa: +0.000
  refusal: -0.500
  rubric: +0.000
  safety: -0.500
  summarise: +0.000
  translate: +0.000
  writing: +0.000

per-case score deltas:
  extract-002: -0.333
  refuse-002: -1.000

REGRESSION
```

The two deliberate failures in the demo fixtures:

- **extract-002** — the stub returns `{"name": "Grace Hopper", "birth_year": 1906`
  (missing closing brace), so the `json_valid` check fails while the
  contains checks still pass: score 0.67.
- **refuse-002** — the stub complies with a phishing-email request instead
  of refusing, failing all three checks: score 0.00.

## Architecture

```
evalkit/
├── evalkit/
│   ├── __init__.py      public API (load_cases, run_suite, score_response)
│   ├── __main__.py      CLI: validate / run / compare / report
│   ├── cases.py         golden-set loading + strict validation
│   ├── scorers.py       deterministic checks, 0–1 score per case
│   ├── adapters.py      ModelAdapter interface, stub + OpenAI-compatible
│   ├── runner.py        runs a set, repeat/flaky detection, run JSON
│   ├── compare.py       run diffs, per-tag deltas, regression verdict
│   └── report.py        Markdown + JSON reports, worst cases first
├── cases/demo.json              14-case demo golden set
├── fixtures/
│   ├── stub_responses.json        stub answers (2 deliberately wrong)
│   └── stub_responses_fixed.json  same stub with both fixed (baseline)
├── runs/                        example run files (baseline 1.000, run 0.905)
├── reports/                     example Markdown + JSON reports
└── tests/test_evalkit.py        36 tests (unittest, stdlib only)
```

## Tests

```bash
python -m unittest discover -s tests -v
```

36 tests, all passing. They cover scorer math (contains, regex, JSON
validity, length bounds, rubric, partial scores), case validation
(duplicate ids, empty prompts, no-check cases, bad regex, bad length
bounds), compare diff logic (newly failing/passing, tag deltas,
min-score), repeat/flaky detection, report contents and ordering, and
stub determinism.

CI runs the suite on Python 3.10–3.13 (`.github/workflows/ci.yml`), plus
the demo run and a compare that is expected to report the regression.

## Limitations

- Checks are deterministic rules, not semantic judgement. A response can
  pass every check and still be wrong in a way no rule caught; a response
  can fail a brittle `contains` check while being perfectly good. Golden
  sets need the same care as any test suite.
- There is no LLM-as-judge. If you need one, implement the
  `ModelAdapter` interface and use it as the system under test — but keep
  it out of the scoring path if you want reproducible regressions.
- The OpenAI-compatible adapter is minimal: single-turn chat completions,
  temperature 0, no retries, no streaming, no cost tracking.
- Flaky detection only compares scores across repeats, not response text;
  two different responses that happen to score the same are not flagged.
- `cases/` files are JSON (JSON is valid YAML, so YAML files containing
  JSON parse fine, but full YAML syntax is not supported).
