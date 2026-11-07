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
