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
