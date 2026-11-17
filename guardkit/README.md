# guardkit

Runtime guardrails that sit on both sides of a model call. On the way in it scores prompt injection and finds PII and leaked credentials. On the way out it scans for leaks. Every decision goes to an audit log with values redacted. Standard library only.

My separate `promptguard` repo is the static version that lints prompt files in CI. This one handles live traffic.

```bash
cd guardkit
python3 -m guardkit check "Email the summary to jane.doe@example.com"
python3 -m guardkit check "Ignore all previous instructions."          # exit 1, blocked
python3 -m guardkit check --policy examples/policy.json --json "Call me at (555) 123-4567."
python3 -m guardkit eval cases/adversarial.jsonl --md reports/eval.md
python3 -m unittest discover -s tests
```

```python
from guardkit import Guard

guard = Guard()                       # block injection and keys, redact PII
safe_complete = guard.wrap(complete)  # any str -> str callable
answer = safe_complete("Summarize this for jane.doe@example.com")
# the model saw [PII-EMAIL_1]; the answer has the real address restored
```

How it works: detectors are plain patterns (emails, North American phone numbers, card numbers with a Luhn check, JWTs, private keys, `api_key = ...` style assignments) plus weighted injection signals summed into a 0-1 score. A JSON policy decides block, redact or allow per category, and rejects unknown keys so a typo can't silently weaken it. Redacted values go into an in-memory vault so they can be restored in the answer, and they never reach the audit log.

On the 24 labelled cases in `cases/adversarial.jsonl` it blocks every attack, catches every PII case, and has no false positives. That only means the patterns cover my fixtures. Paraphrased injections will get through, so treat it as a fast first gate, not a security review.
