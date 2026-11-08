"""Multi-agent pipeline whose agents share one llm-router (offline, simulated models).

Planner needs quality, scrapers are cheap and run in parallel, the writer is
balanced. Prints per-agent spend and how much the routing saved versus
sending every call to the top model.

Model prices/latencies come from llm-router/examples/models.json and are
illustrative, so the savings figure demonstrates the mechanism, not a vendor bill.
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "src"))
sys.path.insert(0, os.path.join(here, "..", "llm-router"))
from agentflow import Agent, Orchestrator, Task, RouterBackend
from llmrouter import Router, SimProvider, ResponseCache, load_models

models = load_models(os.path.join(here, "..", "llm-router", "examples", "models.json"))
router = Router(models, SimProvider(), cache=ResponseCache())
backend = RouterBackend(router)

agents = [
    Agent("planner", backend.llm("planner", strategy="quality")),
    Agent("scraper", backend.llm("scraper", strategy="cheapest", max_tokens=64)),
    Agent("writer", backend.llm("writer", strategy="balanced", max_tokens=300)),
]
tasks = [
    Task("plan", "planner", "Break 'compare vector databases' into three research questions."),
    Task("s1", "scraper", "Collect notes on indexing speed. Context: {plan}", ["plan"]),
    Task("s2", "scraper", "Collect notes on filtering support. Context: {plan}", ["plan"]),
    Task("s3", "scraper", "Collect notes on operating cost. Context: {plan}", ["plan"]),
    Task("draft", "writer", "Write a summary from: {s1} {s2} {s3}", ["s1", "s2", "s3"]),
]
results = Orchestrator(agents, max_workers=3).run(tasks)
for t in tasks:
    tid, r = t.id, results[t.id]
    print(f"{tid:6}{r.status:8}{r.output[:60]}")
print()
print(backend.format_report())
s = router.stats()
print(f"\nrouter cost ${s['total_cost_usd']:.6f} vs all-top-model ${s['baseline_cost_usd']:.6f} "
      f"-> {s['savings_pct']}% saved (simulated)")
