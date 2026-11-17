# llm-router

A router for LLM calls. For each prompt it guesses how hard it is, picks the cheapest (or fastest, or best) model that's good enough, falls back when a model fails, stops hitting dead models with a circuit breaker, caches repeats, and stays inside a spend budget. Standard library only.

Sending everything to the biggest model is the easy default and the expensive one. Most traffic doesn't need it, so I wanted the trade-off to be explicit, with every routing decision coming back with the reasons behind it.

```bash
cd llm-router
python -m llmrouter explain "Design a rate limiter. Compare token bucket and sliding window."
python -m llmrouter simulate --requests 500 --outage pro-l:100:200 --outage open-l:150:250
python -m llmrouter simulate --strategy balanced --budget 0.05 --json
python -m pytest -q
```

```python
from llmrouter import Router, SimProvider, ResponseCache, Budget, load_models

router = Router(load_models("examples/models.json"), SimProvider(),
                strategy="balanced", cache=ResponseCache(), budget=Budget(5.0))
result = router.route("Summarize the causes of the French Revolution.")
print(result.model, result.cost, result.attempts)
```

For a real backend, implement `complete(model, prompt, max_tokens)` and raise `ProviderError` on recoverable failures.

## Simulated run

500 requests with two models going down partway through (seed 0):

| strategy | cost | saved vs always-best | avg latency |
|---|---|---|---|
| cheapest | $0.019 | 98.4% | 488 ms |
| fastest | $0.060 | 94.8% | 279 ms |
| balanced | $0.032 | 97.2% | 371 ms |
| quality | $0.855 | 26.0% | 2131 ms |

The model catalogue in `examples/models.json` is made up (prices, latencies, quality scores), so this shows how the routing behaves, not what any real vendor costs. About a quarter of the workload is repeats, which the cache absorbs.

## Also in here

A cascade mode (cheapest model first, escalate while a verifier rejects the answer), per-model token-bucket rate limits, observed-latency tracking so slow models get routed around, a lexical semantic cache that can be saved to disk, and an optional logistic-regression difficulty scorer you can train on your own traffic.

The default difficulty score is a keyword heuristic, so it will misjudge short, deep questions. That's what the cascade is for. Token counts are estimated at ~4 characters per token.
