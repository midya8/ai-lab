"""Tests for eval metrics math."""

import json
import tempfile
import unittest
from pathlib import Path

from ragkit.chunk import Chunk
from ragkit.eval import GoldenQuestion, load_golden, report_text, run_eval
from ragkit.retrieve import Retriever


def retriever_for():
    # Three chunks: with only two, a common word like "a" would get
    # the same IDF as a distinctive word like "return", which does
    # not reflect a real corpus.
    chunks = [
        Chunk("returns.md",
              "Returns are accepted within 30 days of purchase. "
              "Bring your receipt with you.",
              1, 3),
        Chunk("repairs.md",
              "A standard tune-up costs $89. Most repairs are ready "
              "within 3 business days.",
              1, 3),
        Chunk("shipping.md",
              "A tracking email is sent when an order ships. "
              "Ask a staff member for help.",
              1, 2),
    ]
    return Retriever(chunks)


class TestEvalMetrics(unittest.TestCase):
    def test_perfect_recall_and_precision(self):
        summary = run_eval(
            retriever_for(),
            [GoldenQuestion("How long do I have to return a bike?",
                            ["returns.md"], "within 30 days of purchase")],
            k=5,
        )
        self.assertEqual(summary.mean_recall, 1.0)
        self.assertEqual(summary.mean_precision, 1.0)
        self.assertEqual(summary.passed, 1)
        self.assertEqual(summary.failed_questions, [])

    def test_partial_recall_when_expected_source_missing(self):
        summary = run_eval(
            retriever_for(),
            [GoldenQuestion("How long do I have to return a bike?",
                            ["returns.md", "moon.md"],
                            "within 30 days of purchase")],
            k=5,
        )
        self.assertEqual(summary.mean_recall, 0.5)
        self.assertEqual(summary.passed, 0)
        self.assertEqual(len(summary.failed_questions), 1)

    def test_precision_zero_when_span_not_in_cited_chunks(self):
        summary = run_eval(
            retriever_for(),
            [GoldenQuestion("How long do I have to return a bike?",
                            ["returns.md"], "zebra stripes")],
            k=5,
        )
        self.assertEqual(summary.mean_recall, 1.0)
        self.assertEqual(summary.mean_precision, 0.0)
        self.assertEqual(summary.passed, 0)

    def test_precision_none_when_no_answer_span(self):
        summary = run_eval(
            retriever_for(),
            [GoldenQuestion("return policy?", ["returns.md"], "")],
            k=5,
        )
        self.assertIsNone(summary.results[0].precision)
        self.assertEqual(summary.mean_precision, 0.0)

    def test_report_text_contains_metrics_and_failures(self):
        summary = run_eval(
            retriever_for(),
            [
                GoldenQuestion("How long to return?", ["returns.md"],
                               "within 30 days of purchase"),
                GoldenQuestion("What is the moon made of?", ["moon.md"], "cheese"),
            ],
            k=5,
        )
        text = report_text(summary, k=5)
        self.assertIn("mean recall@5:", text)
        self.assertIn("mean citation precision:", text)
        self.assertIn("passed: 1/2", text)
        self.assertIn("[FAIL]", text)

    def test_summary_to_dict_round_trips(self):
        summary = run_eval(
            retriever_for(),
            [GoldenQuestion("return?", ["returns.md"], "within 30 days")],
            k=5,
        )
        data = json.loads(json.dumps(summary.to_dict()))
        self.assertEqual(data["n_questions"], 1)
        self.assertEqual(data["mean_recall_at_k"], 1.0)

    def test_load_golden_reads_json_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.json"
            path.write_text(json.dumps([
                {"question": "q?", "expected_sources": ["a.md"],
                 "answer_span": "span"}
            ]))
            questions = load_golden(path)
            self.assertEqual(len(questions), 1)
            self.assertEqual(questions[0].question, "q?")

    def test_load_golden_rejects_bad_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "golden.json"
            path.write_text("{not json")
            with self.assertRaises(ValueError):
                load_golden(path)


if __name__ == "__main__":
    unittest.main()
