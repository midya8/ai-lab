# promptlab

Treat prompts like code: version them, A/B them with real statistics, and
block a prompt change in CI when it breaks cases that used to work.
Pure standard-library Python, 94 tests, no API keys needed.

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
`judge` (LLM-as-judge against a rubric) is described [below](#llm-as-judge-checks).
A case passes only if all its checks pass; its score is the fraction passed.
A bad check, a render error or a failing model call fails that case and is
recorded — it never aborts the run.

## LLM-as-judge checks

Some criteria ("polite", "mentions the refund window") are awkward to express as regexes. A `judge` check
asks a second model to grade the output 1-5 against a rubric and passes at `min_score` (default 4):

```json
{"id": "r01", "vars": {"ticket": "Charged twice"}, "checks": [
  {"type": "max_words", "value": 60},
  {"type": "judge", "rubric": "Confirms the refund and says it arrives within \"5 business days\".", "min_score": 4, "votes": 3}]}
```

```bash
promptlab run reply --cases cases.jsonl --judge cmd:./my_judge.sh     # prompt on stdin, JSON reply on stdout
```

- **Strict parsing.** The judge must reply `{"score": 1-5, "reason": "..."}` (a fenced or prose-wrapped object, `Score: 4` or a bare `4/5` also parse). A score outside 1-5, a non-integer, or no number at all FAILS the check; nothing is rounded, clamped or guessed.
- **Never silently skipped.** No `--judge`, a judge exception, or an unparseable reply fails the check with a reason (`judge: scored 2/5 < 4: ...`). Deterministic checks in the same case still run and still count toward the case score.
- **Injection-aware prompt.** The output and case inputs are placed in delimited sections, the grader is told they are untrusted data, and any `<rubric>`, `<input>` or `<response>` tag inside them is defanged so output cannot close its own section.
- **`votes: N`** asks the judge N times and takes the lower median (conservative on ties). Judge calls are not cached, so votes stay independent.

### Calibrate before you trust a judge

A judge is itself a model with its own errors. `calibrate` scores a human-labelled sample with the judge and reports
how well they agree, so "the judge passes it" means something measured:

```bash
promptlab calibrate --labels examples/judge_labels.jsonl --judge rule --min-kappa 0.6   # exit 1 if kappa < 0.6
```

```
Judge calibration on 32 human-scored items (pass = score >= 4)
  agreement   71.9%  (95% CI 54.6%-84.4%)
  kappa       0.44 (moderate)
  spearman    0.70
  bias        -0.09 (judge - human; positive = more lenient)   MAE 0.84
  confusion   both-pass 13, both-fail 10, judge-too-lenient 3, judge-too-harsh 6
```

Cohen's kappa corrects raw agreement for chance (a judge that always says "pass" can look 70% accurate on a mostly-passing set and scores kappa 0).
Leniency (`judge-too-lenient`) is the dangerous direction for a CI gate, so it is reported separately from harshness. Warnings are printed for fewer than 30 items,
single-class labels (kappa undefined) and items the judge could not score (counted as fails).

**About the bundled numbers.** `--judge rule` is a deterministic lexical stand-in (quoted rubric terms present/absent), not an LLM. The 32 labels in `examples/judge_labels.jsonl` were written by the repo author before running the judge and the judge was not tuned against them; they include paraphrases ("five working days") that the lexical judge scores too harshly and keyword-stuffed or negated-term outputs it scores too leniently. The moderate kappa is the expected, honest result: a real LLM judge should be calibrated the same way with a larger, independently labelled sample.

## Pairwise judging and position-bias checks

Asking a judge "which of these two replies is better?" is more sensitive than scoring each alone, but judges
tend to favour a *slot* (often the first) regardless of content. `pairwise` judges every pair **twice, with the
candidates swapped**, and only trusts what survives:

| Both orders say | Verdict | Counted as |
|---|---|---|
| the same candidate wins | that candidate | a win |
| tie | tie | `agree` |
| *different* candidates (the judge chose the same slot twice) | tie | `position` flip, pure bias |
| tie in one order, a winner in the other | tie | `hesitant` |

```bash
promptlab pairwise triage v1 v3 --cases cases.jsonl --rubric 'Gives the "category" as JSON.' \
    --judge cmd:./my_pair_judge.sh --max-inconsistent 0.2        # exit 1 if >20% of pairs flip with order
    --gate --max-loss-rate 0.3   # CI gate: exit 1 if B is significantly worse than A, or A wins >30% of pairs
```

The report gives wins/ties, the win rate of B (ties as half), the share of decisive pairs B wins with a Wilson interval, an exact
sign test on the decisive wins, the consistency rate, and, when pairs flip, whether the flips lean to the first or second slot (exact sign test).
Fewer than 6 decisive pairs is reported as underpowered, not as a result. Without `--rubric`, each case's first `judge` check supplies its rubric.
A pair the judge cannot score (exception, `both`, unparseable) is listed as an error, left out of the counts, and makes the command exit 1.
The judge replies `{"winner": "A" | "B" | "tie", "reason": "..."}`; the grading prompt defangs forged `response_a` / `response_b` / `rubric` / `input` tags in candidate text.

### Multi-judge ensembles

Repeat `--also-judge` to add judges. Every judge rules on every pair in both orders, the ensemble takes a **strict majority**
of the judges' reconciled winners (a split becomes a tie; fewer valid verdicts than a majority quorum is an error and exits 1),
and the report shows each judge alone plus agreement: mean pairwise raw agreement, unanimity, and Fleiss' kappa.

```bash
promptlab pairwise triage v1 v3 --cases cases.jsonl --judge cmd:./gpt.sh --also-judge cmd:./claude.sh --also-judge cmd:./local.sh
```

Demo (`--judge rule --also-judge rule --also-judge rule-biased:0.8:A`, v1 vs v3): the two lexical judges pick v3 on all 60 pairs, the
simulated first-slot-biased judge manages 17 wins and 43 flips, and the majority still says v3, with raw agreement 52%, unanimity 28% and
kappa -0.31. That negative kappa is the known kappa paradox (almost every label is "b", so chance agreement is high and the three
judges' disagreements are all on the biased one), not evidence the ensemble is broken; read it together with raw agreement and the per-judge rows.
The judges here are simulated, so this shows the plumbing and the bias being outvoted, not how real LLM judges behave.

### Audit a pairwise judge against humans

```bash
promptlab pair-audit --labels examples/pair_labels.jsonl --judge rule --min-accuracy 0.75 --max-inconsistent 0.1
```

```
Pairwise judge audit on 24 human-labelled pairs (each judged in both orders)
  reconciled accuracy  79.2%  (95% CI 59.5%-90.8%)
  single-order         79.2% (A first)   79.2% (A second)   swap gain +0.0% vs their mean
  consistency          100.0% of pairs answered identically in both orders; 0 position flip(s)
  first-slot picks     50.0% of decisive picks (a judge that follows content sits near 50%, since each candidate is shown once per slot)
```

Now the same audit on a judge with simulated position bias (`--judge rule-biased:0.5`, which answers "A" regardless of content on half of the prompts):

```
  reconciled accuracy  50.0%  (95% CI 31.4%-68.6%)
  single-order         37.5% (A first)   62.5% (A second)   swap gain +0.0% vs their mean
  consistency          29.2% of pairs answered identically in both orders; 14 position flip(s)
  first-slot picks     84.4% of decisive picks (...)
  position bias        flips lean first (14 first vs 0 second, p=0.000)
```

Two things worth noticing. The biased judge's single-order accuracy depends on which order you happened to use (37.5% vs 62.5%), which is how a position-biased
judge makes an A/B result depend on which prompt you listed first; swapping removes that dependence by turning those pairs into ties. And swapping does **not** raise accuracy:
it makes the failure visible (consistency, first-slot rate, the lean test) instead of letting it pass as signal.

**About the bundled numbers.** The `pairwise` demo in `examples/demo.sh` (v1 vs v3, 60-0) is mechanical: its rubric is keyword-based, so the lexical judge simply sees v3's JSON `category` and v1's "Thanks"; it demonstrates the plumbing, not a finding. `--judge rule` is a lexical stand-in (it scores each candidate with the quoted-term `RuleJudge` and picks the higher), symmetric by construction, so its
position bias is exactly zero and its consistency is 100% by design; its 79.2% reflects 5 of 24 pairs it cannot get right (paraphrases like "five working days", negated
terms like "we never ask for your password"). `rule-biased:p[:A|B]` is a **simulation** of position bias so the audit can be shown catching a known problem; real LLM bias is subtler.
The 24 human labels in `examples/pair_labels.jsonl` were written by the repo author alongside the candidates before any judge was run; they are illustrative, not an independent annotation,
and 24 pairs gives wide intervals. A real judge should be audited with a larger, independently labelled set.

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
| `stats.py` | Wilson, exact McNemar, paired bootstrap, Cohen's kappa, Spearman |
| `compare.py` | A/B verdict, Markdown report, baseline regression gate |
| `models.py` | `RuleModel` / `RuleJudge` offline stand-ins, `CommandModel` adapter |
| `judge.py` | judge prompt + strict reply parser, `judge` check, calibration (agreement, kappa, Spearman, bias) |
| `pairwise.py` | both-orders pairwise judging, reconciliation, A/B summary with sign tests, judge audit vs human preferences, `RulePairJudge` / simulated `PositionBiasedJudge` |

## GitHub Action

`action/action.yml` is a composite action that runs `check` and writes a
Markdown summary (`--summary`, here `$GITHUB_STEP_SUMMARY`) listing regressed,
fixed and new cases. Use it from a checkout of this repo:

```yaml
- uses: actions/checkout@v4
- uses: ./promptlab/action
  with: {name: triage, cases: cases.jsonl, baseline: baseline.json}
```

## Limitations

The bundled judge is a lexical stand-in; a real LLM judge costs one extra call per
judged check (times `votes`) and needs its own calibration. Real models are non-deterministic, so for those, run each case
several times or fix temperature to 0 before trusting a small difference.
