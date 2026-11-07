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
