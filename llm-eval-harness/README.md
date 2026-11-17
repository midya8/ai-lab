# llm-eval-harness

A small eval harness for LLMs that I can run in CI to catch regressions. No dependencies beyond the standard library and requests-free HTTP.

Backends: an offline `rules` backend (deterministic, good for CI), `echo`, and `http` for anything OpenAI-compatible (OpenAI, vLLM, Ollama). Scorers: exact, contains, regex, numeric with tolerance, json_keys, token F1, plus pass@k and an LLM-as-judge scorer.

```bash
cd llm-eval-harness
pip install -r requirements.txt
PYTHONPATH=src python -m evalharness.cli suites/basic.json --backend rules --save base.json
PYTHONPATH=src python -m evalharness.cli suites/basic.json --backend echo --baseline base.json   # exits 1
PYTHONPATH=src python -m evalharness.cli suites/basic.json --backend http \
    --url http://localhost:11434/v1/chat/completions --model llama3
python -m pytest -q
```

Backend failures count as failed cases rather than crashing the run. Comparing against a baseline gives per-case regressions, a Markdown report, and exit code 1 if anything regressed.

Other bits: `--html report.html` for a self-contained report, `--matrix rules,echo` to run one suite across several backends side by side, and a composite GitHub Action:

```yaml
- uses: midya8/ml-from-scratch/llm-eval-harness@main
  with: {suite: suites/basic.json, backend: rules, baseline: baseline.json}
```

This overlaps with [`evalkit`](../evalkit). This one is the pluggable-backend version.
