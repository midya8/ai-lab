import unittest

from llmrouter import LatencyTracker, ModelSpec, Router, SimProvider
from llmrouter.providers import Completion

PROMPT = "Summarise this paragraph about caching."


def spec(name, lat):
    return ModelSpec(name, 0.01, 0.04, lat, 10.0, 0.9, 100000)


class SlowProvider(SimProvider):
    """Reports latency scaled per model, regardless of the catalogue."""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def complete(self, model, prompt, max_tokens):
        c = super().complete(model, prompt, max_tokens)
        return Completion(c.text, c.input_tokens, c.output_tokens,
                          c.latency_ms * self.scale.get(model.name, 1.0))


class TrackerTests(unittest.TestCase):
    def test_neutral_until_min_samples(self):
        t = LatencyTracker(window=5, min_samples=3)
        t.record("a", 100, 400)
        t.record("a", 100, 400)
        self.assertEqual(t.factor("a"), 1.0)
        t.record("a", 100, 400)
        self.assertEqual(t.factor("a"), 4.0)

    def test_median_ignores_single_outlier(self):
        t = LatencyTracker(window=5, min_samples=3)
        for obs in (100, 100, 5000, 100, 100):
            t.record("a", 100, obs)
        self.assertEqual(t.factor("a"), 1.0)

    def test_window_slides_and_clamps(self):
        t = LatencyTracker(window=3, min_samples=3, ceiling=5.0)
        for _ in range(3):
            t.record("a", 10, 1000)
        self.assertEqual(t.factor("a"), 5.0)
        for _ in range(3):
            t.record("a", 100, 50)
        self.assertEqual(t.factor("a"), 0.5)
        self.assertEqual(t.samples("a"), 3)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            LatencyTracker(window=2, min_samples=3)
        t = LatencyTracker()
        t.record("a", 0, 10)
        self.assertEqual(t.samples("a"), 0)


class RouterIntegration(unittest.TestCase):
    def make(self, tracker):
        models = [spec("fast", 100), spec("steady", 150)]
        return Router(models, SlowProvider({"fast": 8.0}), strategy="fastest",
                      latency_tracker=tracker)

    def test_without_tracker_keeps_choosing_nominally_fastest(self):
        r = self.make(None)
        for _ in range(6):
            self.assertEqual(r.route(PROMPT).model, "fast")

    def test_degraded_model_is_routed_around(self):
        r = self.make(LatencyTracker(window=5, min_samples=3))
        picks = [r.route(PROMPT).model for _ in range(8)]
        self.assertEqual(picks[0], "fast")
        self.assertEqual(picks[-1], "steady")
        self.assertIn("observed_latency", r.stats())
        self.assertGreater(r.stats()["observed_latency"]["fast"]["factor"], 5)

    def test_recovery_after_window_refills(self):
        t = LatencyTracker(window=3, min_samples=3)
        for _ in range(3):
            t.record("fast", 100, 800)
        r = Router([spec("fast", 100), spec("steady", 150)], SimProvider(),
                   strategy="fastest", latency_tracker=t)
        self.assertEqual(r.route(PROMPT).model, "steady")
        for _ in range(3):
            t.record("fast", 100, 100)
        self.assertEqual(r.route(PROMPT).model, "fast")


if __name__ == "__main__":
    unittest.main()
