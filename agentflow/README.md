# agentflow
A small multi-agent orchestration framework (stdlib only).

- **Agents**: ReAct-style loop with a tool registry; tool errors are fed back to the model; step limit guards runaway loops.
- **Tasks**: a dependency DAG; upstream outputs are templated into prompts as `{task_id}`.
- **Orchestrator**: topological planning with cycle detection, parallel execution of independent tasks, per-task retries with exponential backoff, and failure cascade (`skipped`).
- **LLM-agnostic**: anything with `.complete(messages) -> str`; `ScriptedLLM` gives deterministic offline tests.

```
python demo.py
python -m pytest
```

## Tracing and supervisor (v0.2)
- `trace_agent(agent, tracer)` records an LLM/tool span (duration, status, error) per call; `tracer.summary()` aggregates per agent and `tracer.to_json()` exports.
- `Supervisor(router_llm, workers, default)` routes each request to the best worker via `{"route": name}`, falling back to the default on an invalid choice.

## Shared memory (v0.3)
`SharedMemory` is a thread-safe key/value store with token-overlap `search` (key matches weigh double), author stamps, and optional atomic JSON persistence (`SharedMemory(path)`). `attach_memory(agent, memory)` gives an agent `remember`, `recall`, `search` and `forget` tools so agents in one Orchestrator run can hand off facts beyond templated task outputs.

## Memory-aware Supervisor
`Supervisor(router_llm, workers, memory=SharedMemory())` searches shared memory for each request, prepends the top `recall_limit` hits to the worker prompt, records the worker's output under `supervisor:<request>` (disable with `remember_results=False`), and returns `memory_used` keys in the result. Without `memory` behaviour is unchanged.

## Circuit breaker
`Supervisor(..., max_attempts=3, breaker=CircuitBreaker(threshold=3, cooldown=30))` stops sending traffic to a worker after `threshold` consecutive failures. Open workers are skipped (reported as `skipped_open`) until `cooldown` seconds pass; then one trial call runs (half-open) and a success closes the breaker while a failure reopens it. If every worker is open, all are tried anyway rather than failing outright.

## Run metrics
`compute_metrics(tasks, results)` returns a `RunMetrics`: total work, critical path (and its task ids), ideal parallelism (work / critical path), and failed/skipped/retry counts. Each `Result` now carries a `duration`.

## Supervisor inside a DAG
`SupervisedAgent(supervisor, name="auto")` wraps a Supervisor as an Agent, so any Orchestrator task addressed to `auto` is routed to the best worker (with memory, fallbacks and circuit breaking). `routing_summary()` counts routes; `decisions` keeps per-call metadata such as `failed_attempts`. A total failure marks the task failed and skips downstream tasks as usual.

## llm-router backend (v0.4)
`RouterBackend(router, max_cost=None)` lets every agent in a pipeline run on one `llm-router` `Router` (duck-typed, so agentflow still has no dependencies). `backend.llm(name, strategy="cheapest"|"quality"|..., max_tokens=, verifier=)` returns a normal agentflow LLM, so a planner can ask for quality while parallel scrapers ask for the cheapest model. Router calls are serialized behind a lock (the router's stats aren't thread-safe), every call is recorded, and `backend.report()` / `format_report()` give cost, cache hits and fallbacks per agent and per model. `max_cost` refuses further calls with `BudgetExhausted` once spend is reached, which the Orchestrator treats as an ordinary task failure. `json_or_text` is a ready-made verifier that rejects malformed protocol messages so a router cascade escalates to a stronger model.

```
python demo_router.py   # planner -> 3 parallel scrapers -> writer; prints spend per agent
```
Demo prices come from llm-router's illustrative catalogue, so the savings figure shows the mechanism, not a vendor bill.

## Persistent run log + resume (v0.5)
`RunLog(path)` is an append-only JSONL journal: `orch.run(tasks, log=RunLog("run.jsonl"))` records every task result (status, output, error, attempts, duration) as it lands, so a crash loses at most the in-flight tasks. Re-run with `resume=True` and tasks that already finished `ok` are replayed from the journal (no LLM call, `Result.resumed=True`) while failed, skipped, or never-started tasks execute.

Replays are keyed by a **fingerprint**: a hash of the task id, agent, prompt template and the fingerprints of its dependencies. Edit one prompt and only that task and its downstream tasks re-run; sibling branches stay cached. A truncated last line from a crash is skipped on read, writes are lock-serialized across worker threads, and `compact()` atomically drops superseded records.

```
python demo_resume.py                     # first run fails midway, the second resumes and finishes
python -m agentflow log run.jsonl         # per-run table: tasks / ok / failed / skipped / resumed / work
python -m agentflow log run.jsonl --compact
```

## Async orchestration (v0.6)
`AsyncOrchestrator(agents, max_concurrency=8, timeout=None)` runs the same `Task` DAG on an asyncio loop. Agents may have a coroutine `llm.complete` and coroutine tools; plain sync ones run in worker threads, so existing agents work unchanged. A semaphore bounds in-flight tasks, `timeout` is per attempt (a timeout is an ordinary failure and can be retried), failures cascade as `skipped`, and `stream()` yields `(task_id, Result)` as each task lands. Abandoning the stream cancels in-flight tasks. `await orch.run(tasks)` or `orch.run_sync(tasks)`. `run`/`stream`/`run_sync` also take `log=RunLog(path), resume=True`: results are journaled as they land (all finished tasks are written before any is yielded, so abandoning the stream loses nothing finished) and tasks whose fingerprint still matches are replayed without a call, same journal format as the sync `Orchestrator`, so a run started sync can be resumed async and back.

## Tool-call checkpointing (`ToolJournal`)

`RunLog` resumes whole tasks; `ToolJournal` goes one level down so a retried or resumed task does
not repeat side effects (emails, charges, tickets) its agent already performed.

```python
from agentflow import AsyncOrchestrator, ToolJournal
orch = AsyncOrchestrator(agents, tool_journal=ToolJournal("tools.jsonl"))
orch.run_sync(tasks)          # orch.tool_stats -> {"replayed": n, "live": m}
```

Successful calls are journaled as (task fingerprint, call index, tool, args) -> observation. On a
retry, calls are matched positionally and replayed while tool and args are identical; at the first
divergence the rest runs live. Failed calls are not journaled. Editing a task prompt (or upstream
tasks) changes its fingerprint and invalidates its entries. Assumption: the agent re-issues the same
early calls; if it does not, replay simply stops. Works with both `AsyncOrchestrator` and the sync `Orchestrator(agents, tool_journal=...)`, and the journal file is shared between them (tests in `tests/test_toolckpt.py`, `tests/test_toolckpt_sync.py`).

Journals can grow forever, so `ToolJournal(path, ttl_s=...)` ignores entries older than the TTL (timestamped on write; legacy entries never expire) and `journal.compact(live_fps=None)` atomically rewrites the file keeping the newest entry per call and dropping expired or dead-fingerprint entries (`tests/test_toolckpt_compact.py`).
