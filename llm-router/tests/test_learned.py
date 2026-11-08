import tempfile
import unittest
from pathlib import Path

from llmrouter import LearnedComplexity, ModelSpec, Router, SimProvider

EASY = ["what is the capital of france", "say hello", "translate cat to spanish",
        "define photosynthesis", "thanks a lot", "what's 2 plus 2", "who is ada lovelace",
        "spell necessary", "hi there", "what is a noun"]
HARD = ["draft a migration plan from monolith to services with rollback strategy",
        "review this pull request for race conditions and propose fixes",
        "write a threat model for our oauth flow and rank mitigations",
        "reconcile these two ledgers and explain every discrepancy",
        "plan a database sharding scheme for ten billion rows",
        "audit this contract clause for liability exposure",
        "propose a fraud detection pipeline with monitoring and rollback",
        "work out why the nightly batch intermittently deadlocks"]
DATA = [(p, 0.0) for p in EASY] + [(p, 1.0) for p in HARD]
HELD = [("define entropy", 0.0), ("hello again", 0.0),
        ("draft a rollback plan for the sharding migration and review risks", 1.0),
        ("audit this pipeline for race conditions and propose fixes", 1.0)]


class LearnedTests(unittest.TestCase):
    def test_untrained_matches_heuristic(self):
        c = LearnedComplexity()
        self.assertEqual(c.score("hello").score, 0.0)

    def test_training_generalises_beyond_heuristic(self):
        c = LearnedComplexity()
        loss = c.fit(DATA)
        self.assertLess(loss, 0.3)
        self.assertEqual(c.accuracy(DATA), 1.0)
        self.assertGreaterEqual(c.accuracy(HELD), 0.75)

    def test_bad_labels_rejected(self):
        with self.assertRaises(ValueError):
            LearnedComplexity().fit([("x", 2.0)])
        with self.assertRaises(ValueError):
            LearnedComplexity().fit([])

    def test_save_load_roundtrip(self):
        c = LearnedComplexity(); c.fit(DATA)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"; c.save(path)
            c2 = LearnedComplexity.load(path)
        self.assertAlmostEqual(c.predict("plan a rollback"), c2.predict("plan a rollback"), places=4)

    def test_router_uses_scorer(self):
        c = LearnedComplexity(); c.fit(DATA)
        models = [ModelSpec("cheap", 0.001, 0.002, 100.0, 10.0, 0.5, 100000),
                  ModelSpec("top", 0.03, 0.06, 100.0, 10.0, 0.95, 100000)]
        r = Router(models, SimProvider(), scorer=c.score)
        self.assertGreater(r.plan("draft a rollback plan for the sharding migration").complexity.score,
                           r.plan("say hello").complexity.score)


if __name__ == "__main__":
    unittest.main()
