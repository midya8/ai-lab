"""Research -> write -> review pipeline with scripted LLMs (offline, deterministic)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from agentflow import *

kb = ToolRegistry(); kb.register("search", lambda q: "LoRA trains low-rank adapters, cutting trainable params ~100x.")
researcher = Agent("researcher", ScriptedLLM([{"tool": "search", "args": {"q": "LoRA"}}, {"final": "LoRA = low-rank adapters."}]), kb)
writer = Agent("writer", ScriptedLLM(["Blog draft: {LoRA = low-rank adapters.}"]))
reviewer = Agent("reviewer", ScriptedLLM(["Approved."]))
res = Orchestrator([researcher, writer, reviewer]).run([
    Task("research", "researcher", "Research LoRA"),
    Task("draft", "writer", "Write about: {research}", ["research"]),
    Task("review", "reviewer", "Review: {draft}", ["draft"]),
])
for k, r in res.items():
    print(f"{k:9}{r.status:8}{r.output}")
