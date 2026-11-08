import json
import os
import tempfile
import unittest

from llmrouter import ManualClock, SemanticCache


class PersistTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "c.json")

    def test_roundtrip_serves_semantic_hit(self):
        a = SemanticCache(clock=ManualClock())
        a.store("How do I reset my password?", 0, {"text": "go to settings"})
        self.assertEqual(a.save(self.path), 1)
        b = SemanticCache(clock=ManualClock(1000.0))
        self.assertEqual(b.load(self.path), 1)
        hit = b.lookup("how can I reset my password", 0)
        self.assertEqual(hit[0], {"text": "go to settings"})

    def test_ttl_survives_restart_via_age(self):
        c1 = ManualClock(500.0)
        a = SemanticCache(ttl_s=100, clock=c1)
        a.store("alpha beta gamma", 0, "v")
        c1.advance(60)
        a.save(self.path)
        c2 = ManualClock(0.0)  # fresh process, unrelated clock origin
        b = SemanticCache(ttl_s=100, clock=c2)
        b.load(self.path)
        c2.advance(30)
        self.assertIsNotNone(b.lookup("alpha beta gamma", 0))
        c2.advance(20)  # total age 110 > 100
        self.assertIsNone(b.lookup("alpha beta gamma", 0))

    def test_expired_not_saved_or_loaded(self):
        c = ManualClock()
        a = SemanticCache(ttl_s=10, clock=c)
        a.store("old entry text", 0, "v")
        c.advance(11)
        self.assertEqual(a.save(self.path), 0)

    def test_missing_file_and_dim_mismatch(self):
        self.assertEqual(SemanticCache().load(self.path), 0)
        SemanticCache(dim=128).save(self.path)
        with self.assertRaises(ValueError):
            SemanticCache(dim=256).load(self.path)

    def test_bad_version_rejected(self):
        with open(self.path, "w") as f:
            json.dump({"version": 9, "dim": 256, "entries": []}, f)
        with self.assertRaises(ValueError):
            SemanticCache().load(self.path)

    def test_max_entries_keeps_newest(self):
        c = ManualClock()
        a = SemanticCache(clock=c)
        for i, w in enumerate(["apple pie", "banana split", "cherry tart"]):
            a.store(w, 0, i)
            c.advance(1)
        a.save(self.path)
        b = SemanticCache(max_entries=2, clock=ManualClock())
        b.load(self.path)
        self.assertEqual(len(b), 2)
        self.assertIsNone(b.lookup("apple pie", 0))
        self.assertEqual(b.lookup("cherry tart", 0)[0], 2)

    def test_atomic_write_leaves_no_tmp(self):
        SemanticCache().save(self.path)
        self.assertEqual(os.listdir(self.d), ["c.json"])


if __name__ == "__main__":
    unittest.main()
