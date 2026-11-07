# agentkit

A dependency-free Python LLM agent runner: standard library only — if
Python is installed, agentkit runs. CLI plus importable library, 46 unit
tests, CI across Python 3.10–3.13.

Built November 2026 as a portfolio project. Not client work, and not
work done for any employer.

## What it does

An agent loop runs a task in steps — plan, call tools, observe results,
decide the next step — until the provider answers in text or a guardrail
stops the run. A task is a small JSON file naming the task, the tools it
may use, and its limits:

```json
{"task": "Add 17 and 25", "max_steps": 10, "tools": ["calculator"]}
```

## Quickstart

No dependencies, no API key, fully offline with the stub provider:

```bash
python -m agentkit validate examples/add-numbers.json
python -m agentkit run examples/add-numbers.json --dry-run
python -m agentkit run examples/add-numbers.json \
  --trace trace.json --report report.txt
```

Expected output for the demo task:

```
status: success
step 1:
  tool: calculator {"expression": "17 + 25"}
  result: 17 + 25 = 42
step 2:
  answer: Done. Result: 17 + 25 = 42
```

Two more examples ship in `examples/`: `search-notes.json` searches
local markdown notes, and `read-file.json` reads a file inside the
allowed directory.

## Providers

A provider is anything with one method,
`complete(messages, tools) -> ProviderResponse` — either a tool call or
a final text answer. Two ship:

- **StubProvider** — deterministic and rule-based, no network. It maps
  arithmetic, file reads, and note searches in the task text to tool
  calls, then answers with the tool result. Used for tests, demos, and
  CI. A scripted mode (a list of responses returned in order) lets
  tests drive exact loop and step-limit behaviour.
- **OpenAICompatibleProvider** — talks to any OpenAI-compatible
  `/chat/completions` endpoint over `urllib`. The API key comes from
  the `AGENTKIT_API_KEY` environment variable only, and only when you
  pick this provider with `--provider openai`. It is never called by
  the test suite, and no key is ever hardcoded or logged.

## Built-in tools

Tools are Python functions declared with a JSON Schema (name,
description, parameters). Schemas are validated before a run, and
arguments are validated against the schema on every call.

- **calculator** — arithmetic (`+ - * / // % **` and parentheses),
  evaluated by walking the AST. Never `eval()`: names, calls, and
  attribute access are rejected.
- **read_file** — reads a text file confined to a base directory.
  Absolute paths, `..` escapes, and symlinks pointing outside the
  directory are rejected; files over 1 MB are refused.
- **search_notes** — substring search over `*.md` files in a notes
  directory, returning file, line number, and matching text.
- **shell** — opt-in only, behind `--allow-shell`. Never registered by
  default, and every execution logs a warning.

## Safety and guardrails

- **Validation first.** Bad task files, unknown tools, duplicate tool
  names, malformed schemas, and bad arguments are all rejected before
  anything runs, with a message naming the problem.
- **Bounded loop.** A run stops after `max_steps` (default 10,
  configurable per task or with `--max-steps`), and stops early when
  the provider repeats an identical tool call — a stuck agent fails
  instead of looping forever.
- **Unknown tools are rejected, not executed.** A provider can only
  call tools in the task's allowlist; anything else ends the run.
- **Size and time limits.** Tool-call arguments are capped (default
  4 KB), and every tool call has a per-step timeout (default 30 s,
  set per task). A timed-out tool fails the run instead of hanging it.
- **Dry run as a plan.** `--dry-run` validates and prints the plan —
  tools available, step limit, provider — without executing anything.
- **No shell by default.** The shell tool exists only when explicitly
  enabled, and warns on every use.

## Exit codes

A scriptable contract: 0 means the run succeeded (or a dry run /
validation passed), 1 means the agent failed, hit the step limit,
looped, or timed out, and 2 means the task or tool definitions are
invalid.

## Reports, not just logs

Every run writes a JSON trace (steps, tool calls, arguments, results,
tokens if the provider reports them, duration, final status) and a
readable text report, via `--trace` and `--report`:

```json
{
  "status": "success",
  "steps": [
    {"step": 1, "tool_call": {"name": "calculator",
     "arguments": {"expression": "17 + 25"}},
     "tool_result": "17 + 25 = 42", "error": null}
  ]
}
```

## Architecture

Six small modules, one job each:

- `model.py` — tasks, tools, tool calls, step records, run results,
  and all validation. No I/O, no network.
- `tools.py` — the registry and the built-in tools (calculator,
  sandboxed file reader, notes search, opt-in shell).
- `providers.py` — the `Provider` protocol, the deterministic stub,
  and the OpenAI-compatible HTTP provider.
- `runner.py` — the agent loop: step limit, loop detection, allowlist
  and argument-size checks, per-step timeouts.
- `report.py` — the JSON trace and the text report.
- `__main__.py` — the CLI only (`run` / `validate`).

The runner is importable, not just a CLI: `Task.from_file` from
`agentkit.model` and `AgentRunner(...).run(task)` from
`agentkit.runner` run a task inside another program and return a
`RunResult` with a `.status` of `"success"`, `"step_limit"`,
`"loop_detected"`, `"timeout"`, or `"failed"` — the same validation,
guardrails, and semantics as the command line, with no second
implementation to keep in sync.

## Tests

```bash
python -m unittest discover -s tests -v
```

46 tests, covering the step-limit stop, loop detection, unknown-tool
rejection, schema and argument validation, dry-run behaviour, trace
and report contents, stub determinism, the sandboxed file reader,
notes search, the opt-in shell tool, and timeout handling. CI runs
them on Python 3.10–3.13 and exercises the demo end to end.
