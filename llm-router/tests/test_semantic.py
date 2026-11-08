import unittest

from llmrouter import ManualClock, ModelSpec, Router, SemanticCache, SimProvider
from llmrouter.semantic import cosine, embed

M = [ModelSpec("m", 0.01, 0.04, 100.0, 10.0, 0.9, 100000)]


class SemanticTests(unittest.TestCase):
    def test_embed_similarity_ordering(self):
        a = embed("How do I reset my password?")
        b = embed("how can i reset my password")
        c = embed("Explain quantum chromodynamics lattice methods")
        self.assertGreater(cosine(a, b), 0.85)
        self.assertLess(cosine(a, c), 0.4)

    def test_empty_prompt_is_safe(self):
        self.assertEqual(embed("   "), {})
        c = SemanticCache()
        c.store("", 0, "x")
        self.assertIsNone(c.lookup("hello world", 0))

    def test_near_duplicate_hits_and_distinct_misses(self):
        c = SemanticCache(threshold=0.85)
        c.store("How do I reset my password?", 0, "v")
        hit = c.lookup("how can I reset my password", 0)
        self.assertEqual(hit[0], "v")
        self.assertEqual(c.semantic_hits, 1)
        self.assertIsNone(c.lookup("What is the capital of France", 0))

    def test_max_tokens_must_match(self):
        c = SemanticCache()
        c.store("reset my password", 100, "v")
        self.assertIsNone(c.lookup("reset my password", 200))

    def test_ttl_and_lru(self):
        clk = ManualClock()
        c = SemanticCache(ttl_s=10, max_entries=2, clock=clk)
        c.store("alpha beta gamma", 0, 1)
        clk.advance(11)
        self.assertIsNone(c.lookup("alpha beta gamma", 0))
        c.store("alpha one", 0, 1)
        c.store("beta two", 0, 2)
        c.store("gamma three", 0, 3)
        self.assertEqual(len(c), 2)
        self.assertIsNone(c.lookup("alpha one", 0))

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            SemanticCache(threshold=0)

    def test_router_serves_paraphrase_from_cache(self):
        r = Router(M, SimProvider(), cache=SemanticCache(threshold=0.85))
        first = r.route("How do I reset my password?")
        second = r.route("how can I reset my password")
        other = r.route("Write a haiku about autumn leaves")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(second.text, first.text)
        self.assertFalse(other.cached)
        self.assertEqual(r.stats()["cache_hits"], 1)


if __name__ == "__main__":
    unittest.main()
