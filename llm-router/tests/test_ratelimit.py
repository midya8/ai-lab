import unittest

from llmrouter import ManualClock, ModelSpec, Router, SimProvider, TokenBucket

PROMPT = "Summarise this paragraph about caching."


def specs():
    return [ModelSpec("cheap", 0.01, 0.04, 100, 10.0, 0.9, 100000),
            ModelSpec("big", 0.5, 1.5, 300, 10.0, 0.95, 100000)]


class BucketTests(unittest.TestCase):
    def test_burst_then_refill(self):
        c = ManualClock()
        b = TokenBucket(1.0, capacity=2, clock=c)
        self.assertTrue(b.try_acquire())
        self.assertTrue(b.try_acquire())
        self.assertFalse(b.try_acquire())
        self.assertAlmostEqual(b.wait_time(), 1.0)
        c.advance(1.0)
        self.assertTrue(b.try_acquire())

    def test_capacity_caps_refill(self):
        c = ManualClock()
        b = TokenBucket.per_minute(3, clock=c)
        c.advance(10_000)
        self.assertEqual(b.tokens, 3.0)

    def test_available_does_not_consume(self):
        b = TokenBucket(1.0, capacity=1, clock=ManualClock())
        self.assertTrue(b.available())
        self.assertTrue(b.available())
        self.assertTrue(b.try_acquire())

    def test_validation(self):
        with self.assertRaises(ValueError):
            TokenBucket(0)
        with self.assertRaises(ValueError):
            TokenBucket(1, capacity=0.5)


class RouterRateLimitTests(unittest.TestCase):
    def setUp(self):
        self.clock = ManualClock()

    def router(self, **kw):
        return Router(specs(), SimProvider(clock=self.clock), clock=self.clock,
                      strategy="cheapest", **kw)

    def test_falls_over_when_bucket_empty(self):
        r = self.router(rate_limits={"cheap": TokenBucket.per_minute(1, clock=self.clock)})
        first = r.route(PROMPT)
        second = r.route(PROMPT + " again")
        self.assertEqual(first.model, "cheap")
        self.assertEqual(second.model, "big")
        self.assertTrue(any("rate limited" in a.note for a in second.attempts))
        self.assertEqual(r.stats()["rate_limited_skips"], 1)

    def test_recovers_after_refill(self):
        r = self.router(rate_limits={"cheap": TokenBucket.per_minute(1, clock=self.clock)})
        r.route(PROMPT)
        self.clock.advance(61)
        self.assertEqual(r.route(PROMPT + " later").model, "cheap")

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            self.router(rate_limits={"nope": TokenBucket(1)})

    def test_cache_hit_spends_no_token(self):
        from llmrouter import ResponseCache
        b = TokenBucket.per_minute(1, clock=self.clock)
        r = self.router(rate_limits={"cheap": b}, cache=ResponseCache(clock=self.clock))
        r.route(PROMPT)
        self.assertTrue(r.route(PROMPT).cached)
        self.assertEqual(r.stats()["rate_limited_skips"], 0)


if __name__ == "__main__":
    unittest.main()
