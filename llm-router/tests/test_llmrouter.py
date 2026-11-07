import json
import tempfile
import unittest
from pathlib import Path

from llmrouter import (AllModelsFailed, Budget, BudgetExceeded, CircuitBreaker,
                       ManualClock, ModelSpec, ProviderError, ResponseCache, Router,
                       RouterError, SimProvider, cache_key, load_models, score_prompt)
from llmrouter.__main__ import main as cli_main
from llmrouter.models import CatalogueError, estimate_tokens
from llmrouter.sim import build_workload, parse_outage, simulate

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "models.json"
HARD = ("Design a rate limiter. Compare and contrast token bucket and sliding window, "
        "ensure that it handles skew, you must optimize p99.")


def spec(name, cost, lat, q, ctx=100000, tpm=10.0):
    return ModelSpec(name, cost, cost * 4, lat, tpm, q, ctx)


def catalogue():
    return [spec("small", 0.001, 100, 0.40), spec("mid", 0.01, 300, 0.70),
            spec("big", 0.1, 900, 0.95)]


class ComplexityTests(unittest.TestCase):
    def test_greeting_is_trivial(self):
        c = score_prompt("Hi there!")
        self.assertEqual(c.score, 0.0)
        self.assertAlmostEqual(c.required_quality(), 0.35)

    def test_hard_prompt_scores_high_with_reasons(self):
        c = score_prompt(HARD)
        self.assertGreaterEqual(c.score, 0.5)
        self.assertTrue(any("reasoning" in r for r in c.reasons))

    def test_code_and_math_raise_score(self):
        base = score_prompt("tell me about lists").score
        self.assertGreater(score_prompt("fix ```def f(): pass```").score, base)
        self.assertGreater(score_prompt("compute 3 * 4 = ?").score, base)

    def test_score_is_clamped(self):
        c = score_prompt(HARD * 200 + "```code``` 2 ^ 3 integral")
        self.assertLessEqual(c.score, 1.0)
        self.assertLessEqual(c.required_quality(), 0.95)


class CatalogueTests(unittest.TestCase):
    def test_example_catalogue_loads(self):
        models = load_models(EXAMPLE)
        self.assertGreaterEqual(len(models), 4)

    def test_cost_math(self):
        m = ModelSpec("x", 1.0, 2.0, 100, 10, 0.5, 1000)
        self.assertAlmostEqual(m.cost(500, 250), 0.5 + 0.5)
        self.assertEqual(m.latency(10), 200)

    def test_rejects_bad_catalogues(self):
        def load(obj):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "m.json"
                p.write_text(json.dumps(obj))
                return load_models(p)
        good = json.loads(EXAMPLE.read_text())["models"][0]
        with self.assertRaises(CatalogueError):
            load({"models": []})
        with self.assertRaises(CatalogueError):
            load({"models": [{**good, "quality": 1.5}]})
        with self.assertRaises(CatalogueError):
            load({"models": [{**good, "colour": "red"}]})
        with self.assertRaises(CatalogueError):
            load({"models": [good, good]})
        bad = dict(good); del bad["latency_ms"]
        with self.assertRaises(CatalogueError):
            load({"models": [bad]})


class CacheTests(unittest.TestCase):
    def test_key_normalises_whitespace_and_case(self):
        self.assertEqual(cache_key("Hello   World "), cache_key("hello world"))
        self.assertNotEqual(cache_key("a", 10), cache_key("a", 20))

    def test_ttl_expiry_and_lru_eviction(self):
        clock = ManualClock()
        c = ResponseCache(max_entries=2, ttl_s=10, clock=clock)
        c.put("a", 1); c.put("b", 2)
        self.assertEqual(c.get("a"), 1)      # a is now most recent
        c.put("c", 3)                         # evicts b
        self.assertIsNone(c.get("b"))
        clock.advance(11)
        self.assertIsNone(c.get("a"))
        self.assertEqual(c.hits, 1)
        self.assertEqual(len(c), 1)  # only "c" left; "a" was dropped on expiry


class BreakerTests(unittest.TestCase):
    def test_opens_then_half_open_then_closes(self):
        clock = ManualClock()
        b = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=clock)
        b.record_failure(); self.assertEqual(b.state, "closed")
        b.record_failure(); self.assertEqual(b.state, "open")
        self.assertFalse(b.allow())
        clock.advance(10)
        self.assertEqual(b.state, "half-open")
        self.assertTrue(b.allow())
        self.assertFalse(b.allow())          # only one probe at a time
        b.record_success()
        self.assertEqual(b.state, "closed")
        self.assertTrue(b.allow())

    def test_failed_probe_reopens(self):
        clock = ManualClock()
        b = CircuitBreaker(1, 5, clock)
        b.record_failure()
        clock.advance(5)
        self.assertTrue(b.allow())
        b.record_failure()
        self.assertEqual(b.state, "open")
        clock.advance(4)
        self.assertEqual(b.state, "open")    # cooldown restarted


class BudgetTests(unittest.TestCase):
    def test_rolling_window_and_soft_limit(self):
        clock = ManualClock()
        b = Budget(1.0, window_s=100, soft_ratio=0.5, clock=clock)
        b.record(0.6)
        self.assertTrue(b.degraded)
        self.assertAlmostEqual(b.remaining, 0.4)
        self.assertFalse(b.can_afford(0.5))
        clock.advance(101)
        self.assertEqual(b.spent, 0.0)
        self.assertFalse(b.degraded)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Budget(0)
        with self.assertRaises(ValueError):
            Budget(1, soft_ratio=0)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.clock = ManualClock()
        self.provider = SimProvider(clock=self.clock)

    def router(self, **kw):
        kw.setdefault("clock", self.clock)
        return Router(catalogue(), self.provider, **kw)

    def test_easy_prompt_goes_to_cheapest(self):
        r = self.router().route("Hi there!")
        self.assertEqual(r.model, "small")
        self.assertEqual(r.attempts[0].outcome, "ok")

    def test_hard_prompt_needs_stronger_model(self):
        r = self.router().route(HARD)
        self.assertIn(r.model, ("mid", "big"))
        self.assertGreaterEqual(r.required_quality, 0.65)

    def test_strategies_order_candidates(self):
        models = [spec("cheap-slow", 0.001, 900, 0.8), spec("pricey-fast", 0.05, 50, 0.8)]
        rt = lambda s: Router(models, self.provider, strategy=s, clock=self.clock)
        self.assertEqual(rt("cheapest").route("Hi").model, "cheap-slow")
        self.assertEqual(rt("fastest").route("Hi").model, "pricey-fast")
        self.assertEqual(rt("quality").route("Hi").model, "cheap-slow")  # tie -> cheaper

    def test_quality_strategy_picks_best(self):
        self.assertEqual(self.router(strategy="quality").route("Hi").model, "big")

    def test_explain_makes_no_calls(self):
        info = self.router().explain(HARD)
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(info["candidates"][0]["model"], "mid")

    def test_context_window_filter(self):
        models = [spec("tiny-ctx", 0.001, 10, 0.9, ctx=10), spec("wide", 0.1, 10, 0.9)]
        r = Router(models, self.provider, clock=self.clock).route("x" * 400)
        self.assertEqual(r.model, "wide")
        with self.assertRaises(RouterError):
            Router([spec("t", 0.1, 10, 0.9, ctx=5)], self.provider).route("x" * 400)

    def test_fallback_on_error(self):
        self.provider.down = {"mid"}
        r = self.router().route(HARD)
        self.assertEqual(r.model, "big")
        self.assertEqual([a.outcome for a in r.attempts], ["error", "ok"])
        self.assertEqual(r.fallbacks, 1)

    def test_all_fail_raises_with_attempt_log(self):
        self.provider.down = {"mid", "big"}
        with self.assertRaises(AllModelsFailed) as ctx:
            self.router().route(HARD)
        self.assertEqual(len(ctx.exception.attempts), 2)

    def test_no_silent_quality_downgrade_by_default(self):
        self.provider.down = {"mid", "big"}
        with self.assertRaises(AllModelsFailed):
            self.router().route(HARD)
        self.assertNotIn("small", self.provider.calls)
        r = self.router(allow_downgrade_on_failure=True).route(HARD)
        self.assertEqual(r.model, "small")

    def test_breaker_stops_hammering_a_dead_model(self):
        self.provider.down = {"mid"}
        router = self.router(breaker_threshold=2, breaker_cooldown_s=60)
        for i in range(5):
            router.route(HARD + f" #{i}")
        self.assertEqual(self.provider.calls.count("mid"), 2)  # then circuit open
        self.provider.down = set()
        self.clock.advance(61)
        r = router.route(HARD + " recovered")
        self.assertEqual(r.model, "mid")                        # half-open probe succeeds
        self.assertEqual(router.breakers["mid"].state, "closed")

    def test_cache_hit_is_free_and_skips_provider(self):
        router = self.router(cache=ResponseCache(clock=self.clock))
        first = router.route("Define entropy.")
        calls = len(self.provider.calls)
        second = router.route("  define   ENTROPY. ")
        self.assertTrue(second.cached)
        self.assertEqual(second.cost, 0.0)
        self.assertEqual(second.text, first.text)
        self.assertEqual(len(self.provider.calls), calls)
        self.assertEqual(router.stats()["cache_hits"], 1)

    def test_budget_degrades_then_blocks(self):
        very_hard = HARD + " ```def f(): pass``` and check 2 * 3 = 6"
        self.assertEqual(self.router().route(very_hard).model, "big")
        budget = Budget(0.01, soft_ratio=0.5, clock=self.clock)
        router = self.router(budget=budget)
        budget.record(0.006)   # past the soft limit
        r = router.route(very_hard + " again")
        self.assertTrue(r.degraded)
        self.assertEqual(r.model, "mid")      # quality floor lowered, cheapest forced
        self.assertEqual(router.stats()["budget_degraded_requests"], 1)
        budget.record(0.01)
        with self.assertRaises(BudgetExceeded):
            router.route("Another question entirely")

    def test_budget_skips_unaffordable_model_and_uses_cheaper_eligible(self):
        models = [spec("a", 0.001, 100, 0.9), spec("b", 5.0, 100, 0.9)]
        budget = Budget(0.5, clock=self.clock)
        r = Router(models, self.provider, strategy="quality", budget=budget,
                   clock=self.clock).route("Hi")
        self.assertEqual(r.model, "a")   # b ties on quality but costs far too much
        self.assertEqual(r.attempts[0].outcome, "ok")

    def test_cascade_escalates_until_verified(self):
        calls = []
        def verifier(text):
            calls.append(text)
            return "[big:" in text
        r = self.router().route("Hi", verifier=verifier)
        self.assertEqual(r.model, "big")
        self.assertTrue(r.verified)
        self.assertEqual([a.outcome for a in r.attempts], ["rejected", "rejected", "ok"])
        self.assertAlmostEqual(r.cost, sum(a.cost for a in r.attempts))

    def test_cascade_all_rejected_returns_unverified_and_uncached(self):
        cache = ResponseCache(clock=self.clock)
        router = self.router(cache=cache)
        r = router.route("Hi", verifier=lambda t: False)
        self.assertIs(r.verified, False)
        self.assertEqual(len(cache), 0)

    def test_stats_savings_vs_always_top_model(self):
        router = self.router()
        for i in range(10):
            router.route(f"What is item {i}?")
        s = router.stats()
        self.assertEqual(s["requests"], 10)
        self.assertGreater(s["savings_pct"], 90)
        self.assertEqual(s["per_model"]["small"], 10)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            self.router(strategy="vibes")
        with self.assertRaises(ValueError):
            self.router().plan("x", strategy="vibes")

    def test_provider_error_type(self):
        self.assertTrue(issubclass(ProviderError, RuntimeError))


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.models = load_models(EXAMPLE)

    def test_workload_is_deterministic(self):
        self.assertEqual(build_workload(50, 3), build_workload(50, 3))
        self.assertNotEqual(build_workload(50, 3), build_workload(50, 4))

    def test_routing_beats_always_top_model(self):
        r = simulate(self.models, "balanced", n=200, seed=1)
        self.assertGreater(r["savings_pct"], 50)
        self.assertEqual(r["unroutable"], 0)

    def test_survives_outage_of_mid_tier(self):
        out = [parse_outage("pro-l:0:200"), parse_outage("open-l:0:200")]
        r = simulate(self.models, "cheapest", n=200, seed=2, outages=out, failure_rate=0)
        self.assertEqual(r["unroutable"], 0)
        self.assertEqual(r["per_model"]["pro-l"], 0)

    def test_tiny_budget_drops_requests_instead_of_overspending(self):
        r = simulate(self.models, "quality", n=100, seed=0, budget_usd=0.01, use_cache=False)
        self.assertLessEqual(r["total_cost_usd"], 0.01 + 0.003)
        self.assertGreater(r["rejected_over_budget"], 0)

    def test_parse_outage(self):
        o = parse_outage("pro-l:10:20")
        self.assertEqual((o.model, o.start, o.end), ("pro-l", 10, 20))
        for bad in ("pro-l", "pro-l:5:5", "x:a:b"):
            with self.assertRaises(ValueError):
                parse_outage(bad)


class CliTests(unittest.TestCase):
    def test_explain_and_simulate_run(self):
        self.assertEqual(cli_main(["explain", "Hi there!"]), 0)
        self.assertEqual(cli_main(["simulate", "--requests", "40", "--strategy", "cheapest"]), 0)

    def test_bad_outage_returns_2(self):
        self.assertEqual(cli_main(["simulate", "--outage", "nope"]), 2)

    def test_missing_catalogue_returns_2(self):
        self.assertEqual(cli_main(["--models", "/nonexistent.json", "explain", "x"]), 2)


if __name__ == "__main__":
    unittest.main()
