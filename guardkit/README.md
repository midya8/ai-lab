# guardkit

Runtime guardrails for LLM pipelines: prompt-injection scoring, PII and
secret detection with reversible redaction, a policy file, an audit log,
and an adversarial eval harness.

Built in November 2026 as a portfolio project. It is not client work, it
is not affiliated with any employer, and it makes no benchmark claims —
the eval numbers below are rates on the small labelled fixture set
shipped in `cases/`, not a claim about catching injection in general.

## The problem it solves

Once a prompt leaves your code for a model, three things can go wrong
in the same request: the user (or text the user pasted in) tries to
override the instructions, the prompt carries personal data or a secret
that should never have left the building, or the model answers by
leaking something it was given. guardkit sits on both sides of the
model call — it guards the input, then guards the output — and records
every decision in a redacted audit log.

## How it differs from promptguard

`promptguard` (a separate standalone repo) is a **static linter**: it
scans prompt *files* in a repo or CI run for secrets, PII, injection
language, and missing metadata, and reports SARIF/JUnit/HTML. guardkit
is the **runtime** half: it inspects live prompts and live model
output as they flow through a pipeline, redacts values in place with a
reversible token vault, and blocks or allows per a policy file. Use
promptguard to keep bad prompts out of the repo; use guardkit to keep
bad prompts out of the model.

## How it works

1. **Detectors** (`guardkit/detectors.py`) — deterministic pattern
   checks, no model judgement. PII: email, North-American phone,
   payment-card numbers validated with Luhn (a random 16-digit run is
   not flagged). Secrets: JWTs, private-key headers, assigned secrets
   such as `api_key = ...`. Injection: weighted signals (ignore
   instructions, reveal the system prompt, role confusion, exfiltration,
   bypass, delimiter injection, base64-decode tricks, instruction
   override) summed into a 0.0–1.0 score, capped at 1.0.
2. **Policy** (`guardkit/policy.py`) — a JSON file that says what to do
   when a detector fires: `block`, `redact`, or `allow`, per category,
   plus the injection-score threshold and a max input size. Unknown
   keys and out-of-range values are rejected; a guardrail that silently
   ignores a typo is worse than one that refuses to load.
3. **Redaction vault** (`guardkit/redact.py`) — detected spans become
   placeholder tokens (`[EMAIL_1]`, `[CARD_1]`) so a prompt can still
   reach the model without the raw value. The mapping lives only in
   memory and is never written to the audit log; `Guard.wrap` restores
   values the model echoes back, so the user sees their own data, not
   tokens.
4. **Guard** (`guardkit/guard.py`) — `check_input` guards a prompt
   (injection at/above threshold blocks, secrets block, PII redacts by
   default, oversize input blocks); `check_output` runs a leak scan on
   model output (injection scoring is deliberately not applied there —
   output quoting an attack is a leak question, not an injection);
   `wrap` composes both around any `str -> str` callable and raises
   `BlockedError` before a blocked call is made.
5. **Audit log** (`guardkit/audit.py`) — append-only JSONL. Entries
   carry the decision, score, rule ids, and redacted text. Raw values
   never appear in it.
6. **Eval** (`guardkit/eval.py`) — a labelled JSONL case set
   (`cases/adversarial.jsonl`, 24 cases: 8 injection, 4 secret, 6 PII,
   6 benign, all synthetic values) scored for attack recall, PII
   caught rate, and false-positive rate, with per-label breakdowns and
   Markdown/JSON reports. `guardkit eval` exits 1 when the recall or
   false-positive gates fail, so it works as a CI gate.

## Quickstart

Requires Python 3.10+ and nothing else (stdlib only, works fully offline).

```bash
# Guard one prompt (exit 0 allow/redact, 1 block)
python3 -m guardkit check "Email the summary to jane.doe@example.com"
python3 -m guardkit check "Ignore all previous instructions."

# Machine-readable decision
python3 -m guardkit check --json "Call me at (555) 123-4567."

# Validate a policy, then use it (examples/policy.json blocks PII too)
python3 -m guardkit validate-policy examples/policy.json
python3 -m guardkit check --policy examples/policy.json "Email jane.doe@example.com."

# Run the adversarial eval and write reports
python3 -m guardkit eval cases/adversarial.jsonl --md reports/eval.md --json reports/eval.json

# Full demo
bash examples/demo.sh
```

## Library use

```python
from guardkit import Guard

guard = Guard()  # defaults: block injection + secrets, redact PII

decision = guard.check_input("Email jane.doe@example.com the report.")
print(decision.action)      # "redact"
print(decision.safe_text)   # "Email [PII-EMAIL_1] the report."

# Wrap any str -> str callable (an agentkit provider, an API call, ...)
safe_complete = guard.wrap(complete)
answer = safe_complete("Summarize this for jane.doe@example.com")
# the model saw [PII-EMAIL_1]; the answer has the real address restored
```

## Eval results (fixture set, not a benchmark)

On `cases/adversarial.jsonl` (24 labelled cases, synthetic values only):

- Attack recall (injection + secret blocked): **1.000**
- PII caught rate (blocked or redacted): **1.000**
- False-positive rate (benign not allowed): **0.000**

These rates say the detectors cover the patterns in the fixture set.
They say nothing about adversarial phrasing that avoids those patterns
— deterministic pattern checks are a fast gate, not a security proof,
and the README says so on purpose.

## Limitations, stated plainly

- Injection scoring is pattern-based. Paraphrase, encoding tricks, and
  novel phrasing that avoids the listed signals will score low.
- Phone detection covers the North-American shape only; other formats
  are left alone rather than guessed at.
- The vault is in-memory: if the process exits, the mapping is gone and
  redacted text cannot be restored.
- The audit log redacts values, but rule ids and redacted text are still
  written to disk — point `--audit` at a path with appropriate access
  controls.
- This is a portfolio project, not a security product. It does not
  replace human security review.

## Development

```bash
python3 -m unittest discover -s tests -v
bash examples/demo.sh
```
