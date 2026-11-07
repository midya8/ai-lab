"""Tests for evalkit. Run from the project root:

    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evalkit import load_cases, run_suite, score_response
from evalkit.adapters import AdapterError, OpenAICompatibleAdapter, StubAdapter, build_adapter
from evalkit.cases import Case, CaseError, filter_cases
from evalkit.compare import compare_runs, format_compare
from evalkit.report import render_json, render_markdown
from evalkit.runner import RunResult

ROOT = Path(__file__).resolve().parent.parent


def make_case(**overrides):
    base = {"id": "t-001", "prompt": "Say hello.", "expected_contains": ["hello"]}
    base.update(overrides)
    return Case(**base)


class FlakyAdapter:
    """Alternates between a good and a bad answer — for flaky detection."""

    name = "flaky"

    def __init__(self):
        self.calls = 0

    def generate(self, prompt, seed=0):
        self.calls += 1
        return "hello world" if self.calls % 2 == 1 else "goodbye world"


class TestScorers(unittest.TestCase):
    def test_contains_pass(self):
        score = score_response(make_case(), "say hello there")
        self.assertTrue(score.passed)
        self.assertEqual(score.score, 1.0)

    def test_contains_fail(self):
        score = score_response(make_case(), "say goodbye")
        self.assertFalse(score.passed)
        self.assertEqual(score.score, 0.0)
        self.assertEqual(score.failed_checks[0].name, "contains:'hello'")

    def test_not_contains(self):
        case = make_case(expected_contains=[], expected_not_contains=["secret"])
        self.assertTrue(score_response(case, "nothing here").passed)
        self.assertFalse(score_response(case, "the secret word").passed)

    def test_regex_match_and_miss(self):
        case = make_case(expected_contains=[], regex=[r"^Subject: .+"])
        self.assertTrue(score_response(case, "Subject: hi").passed)
        self.assertFalse(score_response(case, "no subject").passed)

    def test_json_validity(self):
        case = make_case(expected_contains=[], json_valid=True)
        self.assertTrue(score_response(case, '{"a": 1}').passed)
        self.assertFalse(score_response(case, '{"a": 1').passed)
        self.assertFalse(score_response(case, "not json").passed)

    def test_length_bounds(self):
        case = make_case(expected_contains=[], length_min=3, length_max=5)
        self.assertTrue(score_response(case, "abcd").passed)
        self.assertFalse(score_response(case, "ab").passed)
        self.assertFalse(score_response(case, "abcdef").passed)

    def test_length_min_only(self):
        case = make_case(expected_contains=[], length_min=5)
        self.assertTrue(score_response(case, "12345").passed)
        self.assertFalse(score_response(case, "1234").passed)

    def test_rubric_contains_and_regex(self):
        case = make_case(
            expected_contains=[],
            rubric=[
                {"name": "mentions x", "contains": "x"},
                {"name": "has digits", "regex": r"\d+"},
            ],
        )
        from evalkit.cases import RubricItem

        case.rubric = [RubricItem(name="mentions x", contains="x"),
                       RubricItem(name="has digits", regex=r"\d+")]
        self.assertEqual(score_response(case, "x and 42").score, 1.0)
        self.assertEqual(score_response(case, "x only").score, 0.5)
        self.assertEqual(score_response(case, "nothing").score, 0.0)

    def test_partial_score_math(self):
        case = make_case(
            expected_contains=["apple", "banana", "cherry", "date"],
            expected_not_contains=[],
        )
        score = score_response(case, "apple and banana only")
        self.assertAlmostEqual(score.score, 0.5)
        self.assertFalse(score.passed)

    def test_no_checks_scores_one(self):
        # Scorers stay total even if validation would normally reject this.
        case = Case(id="empty", prompt="hi")
        self.assertEqual(score_response(case, "anything").score, 1.0)


class TestCaseLoading(unittest.TestCase):
    def test_load_demo_set(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        self.assertEqual(len(cases), 14)
        ids = [c.id for c in cases]
        self.assertEqual(len(set(ids)), 14)

    def test_duplicate_ids_rejected(self):
        data = {"version": 1, "cases": [
            {"id": "dup", "prompt": "a", "expected_contains": ["a"]},
            {"id": "dup", "prompt": "b", "expected_contains": ["b"]},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with self.assertRaises(CaseError):
                load_cases(path)
        finally:
            os.unlink(path)

    def test_empty_prompt_rejected(self):
        data = {"version": 1, "cases": [
            {"id": "x", "prompt": "   ", "expected_contains": ["a"]},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with self.assertRaises(CaseError):
                load_cases(path)
        finally:
            os.unlink(path)

    def test_case_with_no_checks_rejected(self):
        data = {"version": 1, "cases": [{"id": "x", "prompt": "hello"}]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with self.assertRaises(CaseError):
                load_cases(path)
        finally:
            os.unlink(path)

    def test_invalid_regex_rejected(self):
        data = {"version": 1, "cases": [
            {"id": "x", "prompt": "hello", "regex": ["([unclosed"]},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with self.assertRaises(CaseError):
                load_cases(path)
        finally:
            os.unlink(path)

    def test_length_min_gt_max_rejected(self):
        data = {"version": 1, "cases": [
            {"id": "x", "prompt": "hello", "length": {"min": 10, "max": 5}},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with self.assertRaises(CaseError):
                load_cases(path)
        finally:
            os.unlink(path)

    def test_missing_file_rejected(self):
        with self.assertRaises(CaseError):
            load_cases(ROOT / "cases" / "does-not-exist.json")

    def test_filter_by_tag_and_id(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        self.assertEqual(len(filter_cases(cases, only_tag="json")), 3)
        self.assertEqual(len(filter_cases(cases, id_substring="extract")), 2)
        self.assertEqual(len(filter_cases(cases, only_tag="json", id_substring="002")), 2)


class TestAdapters(unittest.TestCase):
    def test_stub_determinism(self):
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        prompt = "Summarise the quarterly earnings call in two sentences."
        self.assertEqual(adapter.generate(prompt), adapter.generate(prompt, seed=99))

    def test_stub_default_response(self):
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        self.assertEqual(
            adapter.generate("some unrelated prompt"),
            "I don't have a canned answer for that prompt.",
        )

    def test_stub_rule_matching_is_case_insensitive(self):
        adapter = StubAdapter(rules=[{"match": "Hello", "response": "hi"}])
        self.assertEqual(adapter.generate("say HELLO there"), "hi")

    def test_unknown_adapter_rejected(self):
        with self.assertRaises(AdapterError):
            build_adapter("nope")

    def test_stub_without_fixtures_rejected(self):
        with self.assertRaises(AdapterError):
            build_adapter("stub")

    def test_openai_adapter_needs_key(self):
        env = dict(os.environ)
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(AdapterError):
                OpenAICompatibleAdapter()
        finally:
            os.environ.clear()
            os.environ.update(env)


class TestRunner(unittest.TestCase):
    def test_demo_run_score_and_failures(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        result = run_suite(cases, adapter)
        self.assertEqual(result.passed_count, 12)
        self.assertEqual(result.failed_count, 2)
        failed = {c.case_id for c in result.cases if not c.passed}
        self.assertEqual(failed, {"extract-002", "refuse-002"})
        # 12 cases score 1.0, extract-002 scores 2/3 (bad JSON), refuse-002 0.
        self.assertAlmostEqual(result.overall_score, (12 + 2 / 3) / 14)

    def test_repeat_detects_flaky(self):
        cases = [make_case()]
        result = run_suite(cases, FlakyAdapter(), repeat=2)
        self.assertTrue(result.cases[0].flaky)
        self.assertEqual(result.flaky_ids, ["t-001"])

    def test_repeat_stable_not_flaky(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        result = run_suite(cases, adapter, repeat=3, seed=7)
        self.assertEqual(result.flaky_ids, [])

    def test_run_result_round_trip(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        result = run_suite(cases, adapter)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.json"
            result.save(path)
            loaded = RunResult.load(path)
        self.assertAlmostEqual(loaded.overall_score, result.overall_score)
        self.assertEqual(loaded.passed_count, result.passed_count)


class TestCompare(unittest.TestCase):
    def _two_runs(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        stub = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        good = run_suite(cases, StubAdapter(rules=[
            {"match": "Grace Hopper", "response": '{"name": "Grace Hopper", "birth_year": 1906}'},
            {"match": "phishing email", "response": "I can't help with that request."},
        ], default=""))
        # The "good" adapter above only answers two prompts; build a full
        # passing baseline by patching the demo fixtures instead.
        rules = json.loads((ROOT / "fixtures" / "stub_responses.json").read_text())["rules"]
        fixed = [dict(r) for r in rules]
        for r in fixed:
            if r["match"] == "Grace Hopper":
                r["response"] = '{"name": "Grace Hopper", "birth_year": 1906}'
            if r["match"] == "phishing email":
                r["response"] = "I can't help with that request."
        baseline = run_suite(cases, StubAdapter(rules=fixed))
        bad = run_suite(cases, stub)
        return baseline, bad

    def test_newly_failing_detected(self):
        baseline, bad = self._two_runs()
        result = compare_runs(baseline, bad)
        self.assertEqual(result.newly_failing, ["extract-002", "refuse-002"])
        self.assertEqual(result.newly_passing, [])
        self.assertTrue(result.regressed)

    def test_identical_runs_no_regression(self):
        baseline, _ = self._two_runs()
        result = compare_runs(baseline, baseline)
        self.assertFalse(result.regressed)
        self.assertEqual(result.newly_failing, [])
        self.assertEqual(result.overall_delta, 0.0)

    def test_newly_passing_detected(self):
        baseline, bad = self._two_runs()
        result = compare_runs(bad, baseline)
        self.assertEqual(result.newly_passing, ["extract-002", "refuse-002"])
        self.assertFalse(result.regressed)

    def test_min_score_regression(self):
        baseline, bad = self._two_runs()
        result = compare_runs(baseline, bad, min_score=0.95)
        self.assertTrue(result.regressed)
        result_ok = compare_runs(baseline, bad, min_score=0.5)
        # Still regressed: two cases newly failed, even though min-score passes.
        self.assertTrue(result_ok.regressed)

    def test_tag_deltas(self):
        baseline, bad = self._two_runs()
        result = compare_runs(baseline, bad)
        self.assertLess(result.tag_deltas["json"], 0)
        self.assertLess(result.tag_deltas["safety"], 0)
        self.assertEqual(result.tag_deltas["qa"], 0)

    def test_format_compare_mentions_regression(self):
        baseline, bad = self._two_runs()
        text = format_compare(compare_runs(baseline, bad))
        self.assertIn("extract-002", text)
        self.assertIn("REGRESSION", text)


class TestReport(unittest.TestCase):
    def test_markdown_worst_first(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        result = run_suite(cases, adapter)
        md = render_markdown(result)
        self.assertIn("## Cases (worst first)", md)
        self.assertIn("extract-002", md)
        self.assertIn("refuse-002", md)
        # Worst case (score 0.00) appears before a passing case in the table.
        self.assertLess(md.index("refuse-002"), md.index("qa-001"))
        self.assertIn("Overall score", md)

    def test_json_report_is_valid(self):
        cases = load_cases(ROOT / "cases" / "demo.json")
        adapter = StubAdapter.from_file(ROOT / "fixtures" / "stub_responses.json")
        result = run_suite(cases, adapter)
        data = json.loads(render_json(result))
        self.assertEqual(data["passed_count"], 12)
        self.assertEqual(data["failed_count"], 2)


if __name__ == "__main__":
    unittest.main()
