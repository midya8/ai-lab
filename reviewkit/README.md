# reviewkit

A pull-request review bot that is honest about what it knows. A deterministic rule pass (AST + tokenizer-aware regex) catches what can be caught reliably, an **optional** LLM pass adds judgement, and everything the model says is validated against the diff before it can be posted. Runs offline by default; a model is opt-in.

```bash
git diff main...HEAD | PYTHONPATH=src python -m reviewkit review --diff - --root . --fail-on high
PYTHONPATH=src python -m reviewkit rules                      # list rules
PYTHONPATH=src python -m reviewkit bench bench/dev_cases.py   # score the rules
PYTHONPATH=src python examples/demo.py                        # no-network end-to-end demo
```

## What it does

- **Diff-aware.** A unified-diff parser tracks new-file line numbers (renames, deletions, binaries, `\ No newline`, plain `diff -u`). Rules only report on lines the PR *added*: risky code that was already there, or merely moved within a hunk context, is not blamed on the author (but see Limitations for moved code).
- **25 rules** (`reviewkit rules`): `eval`/`exec`, `shell=True`, pickle-style deserialisation (including `from pickle import loads`), `torch.load` without `weights_only` (multi-line calls handled), `yaml.load`, `verify=False`, JWT decode without signature check, Hugging Face `trust_remote_code=True`, `allow_dangerous_deserialization=True`, weak hashes, SQL built by f-string / `%` / `+` / `.format`, committed secrets (AWS, GitHub, `sk-`, Slack, private keys, credential-looking assignments), mutable defaults (incl. lambdas), bare/swallowed `except`, `requests` without `timeout` (including `timeout=None`), `is` against literals (including `-1`), cyclomatic complexity, leftover debuggers, merge-conflict markers, syntax errors, and JS `eval` / `innerHTML` / `dangerouslySetInnerHTML`.
- **No false alarms from text.** Python rules run on a tokenizer-built *code view* with strings, docstrings and comments blanked, so `model.eval()`, `"eval(x)"`, `# exec(cmd)` and `def eval(self)` are clean while `f"{eval(q)}"` is still caught.
- **Secrets are never echoed.** The comment says what kind of secret it is and tells you to rotate it; the value is not repeated into the PR thread. Placeholders (`changeme`, `your-api-key`, `${VAR}`), env reads and low-entropy values are skipped; assignment-style secrets are skipped in test paths, token formats are not.
- **LLM pass you can trust the shape of.** The diff is fenced with a random nonce and declared untrusted data. The reply must parse as JSON; each comment must point at a line the PR added in *that chunk* (hallucinated lines are dropped and counted); severity is capped at `high`; `@mentions` are defused and `<!-- -->` is stripped so a model (or an injected diff) cannot ping people or forge reviewkit's own markers. Comments that restate a rule finding on the same line are deduplicated. A failing model call is a note, never a failed review. Providers: any OpenAI-compatible endpoint (retry/backoff on 429/5xx, tested against a local HTTP server), a local command, or a scripted provider for tests.
- **Idempotent posting.** Every inline comment embeds a fingerprint of `(file, rule, line text)`, not the line number. Pushing again only posts new findings, even when earlier code shifted. If GitHub answers 422 (a line outside the diff view), the review falls back to a body-only review instead of vanishing.
- **Control.** `.reviewkit.json`: disable rules, override severities, ignore globs, `max_comments`, `min_severity`, complexity threshold. Inline `# reviewkit: ignore` or `# reviewkit: ignore[rule-id]` (counted, and reported as suppressed). `--fail-on SEVERITY` gates the build.

## GitHub Action

`action.yml` is a composite action; `examples/workflow.yml` is a ready workflow (checkout the PR head with `fetch-depth: 0`, `pull-requests: write`). Inputs reach the script through environment variables, not string interpolation. The action itself has **not** been run on GitHub in this repo (it would need a pull request in a second repo); the CLI path it wraps, the review payload and the posting logic are all covered by tests with a fake GitHub API. Pull requests from forks get a read-only token, so there the job can fail the build but not comment.

## Measured, with the caveats

A rule engine tuned on its own examples proves nothing, so there are two label sets (`bench/`, labels written as `# EXPECT: rule` on the line that should be flagged; before/after files are turned into real unified diffs, then reviewed by the same code path as a PR; any unmarked finding counts as a false positive).

| Set | Cases | TP | FP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| dev (rules tuned on it) | 20 | 38 | 0 | 0 | 1.000 | 1.000 |
| held-out, first run | 14 | 16 | 5 | 8 | 0.762 | 0.667 |
| held-out, after fixing 3 of my own labelling mistakes | 14 | 18 | 3 | 6 | 0.857 | 0.750 |
| held-out, after fixing the real misses | 14 | 24 | 2 | 0 | 0.923 | 1.000 |

The honest reading: the held-out set was written after the rules were frozen, and it found real gaps (`from pickle import loads`, `timeout=None`, `idx is -1`, lambda defaults, `innerHTML +=`, `.format()` into a variable, an over-eager SQL f-string match on prose like `update config set to ...`). I fixed those, which means **the last row is no longer held-out**: those cases are now tuned-on, and a fresh set is needed to say anything new. Of the labelling mistakes, one was a marker on the wrong line of a multi-line call, one used a marker syntax the harness does not read, and one expectation (flagging the import line as well as the call) I dropped because one comment per problem is better. Dogfooding the rules over the other 16 projects in this monorepo (diff against the empty tree, `--unified=0`) found a real false positive (`def eval(self)` read as a call), fixed and added to the dev set, and otherwise 7 test-fixture "secrets" that are fake on purpose, 1 deliberate opt-in `shell=True` in agentkit's shell tool, 4 swallowed exceptions and 71 complexity notes.

## Limitations (known, not hidden)

- Rules are syntactic. `cur.execute("... IN (" + marks + ")", ids)` (a built placeholder list, which is safe) is flagged as SQL injection; there is no taint tracking.
- A function moved within a file shows up as added lines, so risky code that merely moved is flagged (`refactor-moves-existing-risk` in the held-out set).
- AST rules need the PR head checked out under `--root`; with only a diff, line rules still run and a note says the rest was skipped. A working tree that does not match the diff is detected and skipped rather than guessed at.
- JS/TS support is three line rules, no parser. Other languages get only the secret and merge-conflict rules.
- The model pass is only as good as the model; reviewkit bounds what a bad reply can do (no fake lines, no pings, no `critical`), not whether it is right. The bundled demo uses a scripted model.
- Rule severities are opinions; override them in `.reviewkit.json`.

## Layout

```
src/reviewkit/  diff.py rules.py llm.py review.py github.py bench.py cli.py
bench/          dev_cases.py heldout_cases.py
examples/       demo.py workflow.yml reviewkit.json
tests/          73 tests: parser, every rule family, LLM validation, HTTP provider vs a local server,
                dedupe/caps/config, idempotent posting with a fake GitHub, CLI incl. real `git diff`
```

```bash
pip install -r requirements.txt && PYTHONPATH=src python -m pytest -q
```


## Moved-code detection

A block removed in one place and added verbatim (ignoring indentation) elsewhere in the same diff is treated as a move: findings on those lines already existed, so they are skipped and a note says how many. Runs of fewer than `min_move_lines` (default 3) meaningful lines, and trivial lines like lone brackets, never count. Config keys: `skip_moved` (default true), `min_move_lines`. A moved block that is also edited is reviewed as new code (conservative). Matching is exact-line, so a move with a one-line tweak in the middle is split into shorter runs, and runs under the minimum are reviewed normally. Tested on synthetic diffs only, not yet on real PRs.

## Taint rules (py-taint-sql / py-taint-shell / py-taint-eval)

`taint.py` follows values from `input()`, `sys.argv` and `request.args/form/json/...` through assignments, f-strings, `+`, `%` and `.format` into SQL `execute` query text, `os.system`, `subprocess.*(shell=True)` and `eval`/`exec`. `int()`/`float()`/`shlex.quote` clear taint; parameterised queries are clean. Limits: per-function only, function parameters are not assumed tainted, no inter-procedural flow, so it will miss injections that cross function boundaries.
