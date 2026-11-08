# llm-eval-harness
Small, dependency-free harness for evaluating LLMs and catching regressions in CI.

- Pluggable backends: offline `rules` (deterministic, CI-safe), `echo`, and `http` for any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama)
- Scorers: exact, contains, regex, numeric (tolerance), json_keys, token_f1
- JSON prompt suites with templates, per-case scorers and tags
- Threaded runner; backend failures count as failed cases, not crashes
- Baseline comparison: per-case regressions, Markdown report, exit code 1 on regression

```
pip install -r requirements.txt
PYTHONPATH=src python -m evalharness.cli suites/basic.json --backend rules --save base.json
PYTHONPATH=src python -m evalharness.cli suites/basic.json --backend echo --baseline base.json  # exits 1
PYTHONPATH=src python -m evalharness.cli suites/basic.json --backend http --url http://localhost:11434/v1/chat/completions --model llama3
python -m pytest
```

## pass@k and LLM-as-judge
`evalharness.pass_at_k(n, c, k)` is the unbiased estimator from the Codex paper; `evaluate_pass_at_k(backend, cases, scorer, n=5, ks=(1,5))` samples each prompt n times. `judge_scorer(judge_backend, rubric)` returns a scorer that has a second model grade 1-5, mapped to 0..1.

## GitHub Action
`action.yml` is a composite action: it runs a suite, writes the Markdown report to the job summary, saves the run JSON, and fails the job on regression.
```yaml
- uses: midya8/ai-lab/llm-eval-harness@main
  with: {suite: suites/basic.json, backend: rules, baseline: baseline.json}
```
CLI flag `--summary-file PATH` appends the report to any file.

## HTML report
`--html report.html` writes a self-contained page (score bars, per-tag table, latency, regressions highlighted when `--baseline` is given). All model output is HTML-escaped.

## Multi-model matrix
`--matrix echo,rules` runs the suite on several backends and prints a case x model score table with per-model means, the winner, and how many cases the models disagree on. Library: `run_matrix(backends, suite)`, `matrix_table`, `render_matrix_markdown`. Backend names must be unique.
