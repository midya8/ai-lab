# llm-router

A cost- and latency-aware router for LLM calls. For each prompt it scores
difficulty, picks the cheapest (or fastest, or best) model that is good
enough, falls back when a model fails, stops hammering dead models with a
circuit breaker, caches repeats, and enforces a spend budget. Pure
standard-library Python, 39 tests, no API keys needed.

Built in November 2026 as a portfolio project. The model catalogue in
`examples/models.json` is **illustrative** (made-up prices, latencies and
quality scores), and the simulator numbers below are properties of that
synthetic catalogue and workload — not benchmarks of any real vendor.

## Why

Sending every request to the biggest model is the easy default and the
expensive one. Most traffic (greetings, lookups, rewrites) does not need
it. A router makes the trade-off explicit and auditable: every decision
comes with the reasons that produced it.

## How it works

| Piece | File | What it does |
|---|---|---|
| Complexity scoring | `complexity.py` | Transparent heuristic (length, reasoning cues, code, math, constraints, multi-part lists) -> 0..1 score -> minimum model quality needed. Returns the reasons. |
| Strategies | `router.py` | `cheapest`, `fastest`, `quality`, `balanced` (min-max-normalised weighted cost/latency) over the models that clear the quality bar and fit the context window. |
| Fallback chain | `router.py` | On `ProviderError`, try the next candidate. Never silently downgrades below the quality bar unless `allow_downgrade_on_failure=True`. |
| Circuit breaker | `breaker.py` | Per-model closed/open/half-open; after N failures the model is skipped until a cooldown elapses, then exactly one probe is admitted. |
| Cache | `cache.py` | Whitespace/case-normalised exact match, LRU + TTL. Hits cost nothing and still count toward savings. |
| Budget | `budget.py` | Rolling-window USD limit. Past the soft limit the router lowers the quality floor and forces `cheapest`; models that cannot fit in what is left are skipped; nothing affordable raises `BudgetExceeded`. |
| Cascade | `router.py` | Pass `verifier=fn`: cheapest model first, escalate while the verifier rejects, cost of every attempt is counted. Unverified results are never cached. |
| Simulator | `sim.py` | Deterministic workload (50% simple / 30% medium / 20% hard + repeats), injectable outages and budgets, fake clock — runs in milliseconds. |

All time-dependent parts take an injected clock (`ManualClock`), so TTLs,
cooldowns and budget windows are tested without sleeping.

## Usage

```bash
# Why would this prompt go to that model? (no model call)
python -m llmrouter explain "Design a rate limiter. Compare and contrast token bucket and sliding window, ensure that it handles skew, you must optimize p99."

# Replay 500 requests under every strategy, with two models going down mid-run
python -m llmrouter simulate --requests 500 --outage pro-l:100:200 --outage open-l:150:250

# One strategy under a hard $0.05 budget
python -m llmrouter simulate --strategy balanced --budget 0.05 --json
```

Sample `simulate` output (seed 0, 500 requests, two outages):

```
strategy      cost $  baseline $   saved   avg ms  cache  fallbk  dropped
-------------------------------------------------------------------------
cheapest      0.0186      1.1553   98.4%      488    125      16        0
fastest       0.0604      1.1553   94.8%      279    125      15        0
balanced      0.0324      1.1553   97.2%      371    125      15        0
quality       0.8545      1.1553   26.0%     2131    125       9        0
```

"saved" is relative to sending every request to the highest-quality model,
and includes cache hits (about a quarter of this synthetic workload is
repeats). `quality` is the control: it shows what routing buys you.

As a library:

```python
from llmrouter import Router, SimProvider, ResponseCache, Budget, load_models

router = Router(load_models("examples/models.json"), SimProvider(),
                strategy="balanced", cache=ResponseCache(), budget=Budget(5.0))
result = router.route("Summarize the causes of the French Revolution.")
print(result.model, result.cost, result.attempts)
print(router.stats())
```

To use a real backend, implement `complete(model, prompt, max_tokens) ->
Completion` and raise `ProviderError` on recoverable failures; the router
needs nothing else.

## Limits (honest ones)

* The complexity score is a keyword/shape heuristic, not a learned
  classifier. It is easy to audit and tune but will misjudge short,
  deep questions. The cascade mode (`verifier=`) is the safety net.
* Token counts are estimated at ~4 characters per token.
* `ResponseCache` is exact-match after normalisation; `SemanticCache` is opt-in and uses hashed lexical vectors, not neural embeddings, so it catches rephrasings with shared words but not true synonyms.
* `SimProvider` answers are placeholders; this repo measures routing
  behaviour, not answer quality.

## Tests

```bash
python -m pytest        # or: python -m unittest discover -s tests
```

## Observed-latency tracking

Catalogue latencies are static guesses. Pass a `LatencyTracker` and the router learns
the ratio of observed to predicted latency per model (sliding window, median, so one
outlier is ignored; neutral until `min_samples`). `fastest` and `balanced` then route
around a model that is currently slow and return to it once it recovers.

```python
from llmrouter import LatencyTracker, Router
router = Router(models, provider, strategy="fastest",
                latency_tracker=LatencyTracker(window=20, min_samples=3))
router.stats()["observed_latency"]   # {"fast": {"samples": 5, "factor": 8.0}, ...}
```

## Semantic cache

`SemanticCache` serves near-duplicate prompts ("How do I reset my password?" vs
"how can I reset my password") from one entry. Dependency-free hashed word + char-trigram
vectors, cosine threshold (default 0.85), same `max_tokens` required, LRU + TTL.

```python
from llmrouter import Router, SemanticCache
router = Router(models, provider, cache=SemanticCache(threshold=0.85))
```
Lower thresholds raise hit rate but risk serving a wrong answer; tune on your traffic.

Persistence: `cache.save("cache.json")` (atomic, JSON, live entries only) and `cache.load("cache.json")` at startup. Entries keep their *age*, so TTLs stay correct across restarts even though the default clock is monotonic.

## Learned complexity classifier

`LearnedComplexity` is a stdlib logistic regression over hashed word/bigram features, with the
heuristic score as a prior feature. Train it on (prompt, difficulty) pairs from your own traffic
(1 = needed the top model) and plug it in; untrained, it falls back to the heuristic.

```python
from llmrouter import LearnedComplexity, Router
clf = LearnedComplexity(); clf.fit([("say hello", 0.0), ("plan a sharding migration", 1.0)])
clf.save("complexity.json")            # LearnedComplexity.load("complexity.json")
router = Router(models, provider, scorer=clf.score)
```
Needs a reasonable number of labelled prompts to beat the heuristic; the unit-test data is tiny and synthetic.
