# agentkit

A small LLM agent runner with no dependencies beyond the standard library. You give it a task as a JSON file, it loops through plan, call a tool, look at the result, until the model answers in text or a guardrail stops it.

```json
{"task": "Add 17 and 25", "max_steps": 10, "tools": ["calculator"]}
```

```bash
cd agentkit
python -m agentkit validate examples/add-numbers.json
python -m agentkit run examples/add-numbers.json --dry-run
python -m agentkit run examples/add-numbers.json --trace trace.json --report report.txt
python -m unittest discover -s tests
```

```
status: success
step 1:
  tool: calculator {"expression": "17 + 25"}
  result: 17 + 25 = 42
step 2:
  answer: Done. Result: 17 + 25 = 42
```

It runs offline with a deterministic stub provider. `--provider openai` talks to any OpenAI-compatible endpoint, with the key read from `AGENTKIT_API_KEY`.

Built-in tools: a calculator that walks the AST (never `eval`), a file reader confined to one directory, a markdown notes search, and a shell tool that only exists if you pass `--allow-shell`.

The guardrails were the part I cared about most:

- Everything is validated before anything runs.
- Hard step limit.
- Stops if the model repeats the exact same tool call.
- Only allowlisted tools can be called.
- Argument size cap and a per-step timeout.

Exit codes are 0 for success, 1 if the agent failed, looped or timed out, and 2 for an invalid task. `AgentRunner(...).run(task)` gives you the same thing from Python.
