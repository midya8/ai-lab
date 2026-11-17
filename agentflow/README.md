# agentflow

A small multi-agent orchestration framework, standard library only. Agents are ReAct-style loops with a tool registry. Tasks form a dependency DAG, and the orchestrator runs independent tasks in parallel with retries, backoff, and downstream tasks skipped when something upstream fails. Any object with `.complete(messages) -> str` works as the LLM. `ScriptedLLM` keeps the tests offline and deterministic.

```bash
cd agentflow
python demo.py
python demo_resume.py           # first run fails midway, second run resumes
python demo_router.py           # planner -> 3 parallel scrapers -> writer, cost per agent
python -m pytest -q
```

Things I added over time, roughly in order:

- Tracing spans per LLM and tool call, and a `Supervisor` that routes requests to the right worker.
- `SharedMemory`, a thread-safe key/value store agents can read and write through tools, so facts can be handed off between tasks.
- A circuit breaker on the supervisor, so a worker that keeps failing gets skipped for a cooldown.
- Run metrics: total work, critical path, ideal parallelism.
- A backend for [`llm-router`](../llm-router), so each agent can ask for "cheapest" or "quality" and you get a cost report per agent, with an optional hard budget.
- `RunLog`, an append-only JSONL journal. Rerunning with `resume=True` replays finished tasks. Replays are keyed on a fingerprint of the prompt and its upstream tasks, so editing one prompt only reruns that branch.
- `AsyncOrchestrator`, with per-attempt timeouts and streaming results. It shares the journal format, so a run started sync can resume async and the other way round.
- `ToolJournal`, which journals individual tool calls so a retried task doesn't repeat side effects like sending an email twice.

The demo prices come from llm-router's made-up catalogue, so the savings show the mechanism, not a real bill.
