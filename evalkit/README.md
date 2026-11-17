# evalkit

A small regression harness for prompts and models, standard library only. Prompt edits and model upgrades break things quietly: the summary stops mentioning the key number, the JSON extractor drops a brace, a refusal stops holding. evalkit runs a versioned golden set with plain pass/fail rules and diffs the result against the last good run.

```bash
cd evalkit
python -m evalkit validate cases/demo.json
python -m evalkit run cases/demo.json --adapter stub --fixtures fixtures/stub_responses.json \
    --out runs/run.json --repeat 2 --seed 42
python -m evalkit report runs/run.json --md reports/run.md
python -m evalkit compare runs/baseline.json runs/run.json --min-score 0.85   # exits 1 on regression
python -m unittest discover -s tests
```

A case is a prompt plus checks: must contain, must not contain, regex, valid JSON, length bounds, named rubric items. A case's score is the fraction of checks it passes. There's deliberately no LLM-as-judge in the scoring path, because that makes runs non-deterministic, which defeats the point. Adapters are a stub that replays canned responses and an OpenAI-compatible one (key from the environment).

The demo set has 14 cases with two deliberate failures in the stub fixtures: a JSON answer missing its closing brace, and a phishing request the stub goes along with. `compare` against the fixed baseline drops from 1.000 to 0.905, names exactly those two cases, and shows the score change per tag.

Rules aren't semantic judgement. A response can pass every check and still be wrong. Flaky detection only compares scores across repeats, not the text. See also [`llm-eval-harness`](../llm-eval-harness) for the variant with pluggable backends.
