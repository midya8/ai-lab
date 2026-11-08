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
