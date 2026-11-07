"""End-to-end CLI tests (subprocess, offline)."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "-m", "ragkit", *args],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.index = Path(self.tmp.name) / "index.json"
        result = run_cli("index", str(ROOT / "examples" / "docs"), str(self.index))
        self.assertEqual(result.returncode, 0, result.stderr)

    def tearDown(self):
        self.tmp.cleanup()

    def test_ask_returns_cited_answer(self):
        result = run_cli("ask", str(self.index),
                         "How long do I have to return a bike?")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("30 days", result.stdout)
        self.assertIn("[returns.md:", result.stdout)

    def test_ask_tfidf_scoring_works(self):
        result = run_cli("ask", str(self.index),
                         "What does a standard tune-up cost?",
                         "--scoring", "tfidf")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("$89", result.stdout)

    def test_ask_no_results_exits_1(self):
        result = run_cli("ask", str(self.index), "zxqv wjorp flimflam")
        self.assertEqual(result.returncode, 1)
        self.assertIn("No relevant documents", result.stdout)

    def test_ask_empty_query_exits_2(self):
        result = run_cli("ask", str(self.index), "   ")
        self.assertEqual(result.returncode, 2)

    def test_ask_missing_index_exits_2(self):
        result = run_cli("ask", str(self.tmp.name) + "/missing.json", "hello")
        self.assertEqual(result.returncode, 2)

    def test_eval_reports_metrics_and_writes_json(self):
        out_json = Path(self.tmp.name) / "eval.json"
        result = run_cli("eval", str(self.index),
                         str(ROOT / "eval" / "golden.json"),
                         "--json", str(out_json))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mean recall@5:", result.stdout)
        data = json.loads(out_json.read_text())
        self.assertEqual(data["n_questions"], 8)
        self.assertGreaterEqual(data["mean_recall_at_k"], 0.8)

    def test_eval_fail_below_threshold_exits_1(self):
        result = run_cli("eval", str(self.index),
                         str(ROOT / "eval" / "golden.json"),
                         "--fail-below", "1.01")
        self.assertEqual(result.returncode, 1)

    def test_eval_missing_golden_exits_2(self):
        result = run_cli("eval", str(self.index),
                         str(self.tmp.name) + "/missing.json")
        self.assertEqual(result.returncode, 2)

    def test_index_missing_docs_dir_exits_2(self):
        result = run_cli("index", str(self.tmp.name) + "/nope",
                         str(self.tmp.name) + "/out.json")
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
