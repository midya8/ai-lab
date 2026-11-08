import io
import json
import math
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promptlab import (Registry, RegistryError, Template, TemplateError, RuleModel, CommandModel,
                       Case, load_cases, run_check, run_prompt, compare_runs, check_regression,
                       render_markdown, wilson, mcnemar_exact, paired_bootstrap, make_model)
from promptlab.__main__ import main
from promptlab.runner import RunResult, CaseResult

EX = Path(__file__).resolve().parents[1] / "examples"


class MultiSampleTests(unittest.TestCase):
    def _v(self):
        with tempfile.TemporaryDirectory() as d:
            reg = Registry(d)
            v, _ = reg.commit("p", "Say {{x}}", "")
            return v

    def test_flaky_model_detected(self):
        v = self._v()
        seq = iter(["yes", "no", "yes", "yes"])
        cases = [Case("a", {"x": "1"}, [{"type": "contains", "value": "yes"}])]
        r = run_prompt(v, cases, lambda p: next(seq), samples=4)
        c = r.results[0]
        self.assertEqual((c.samples, c.pass_count, c.passed, c.flaky), (4, 3, False, True))
        self.assertAlmostEqual(c.score, 0.75)
        self.assertEqual(r.flaky, ["a"])
        self.assertTrue(c.failures)

    def test_threshold_allows_partial(self):
        v = self._v()
        seq = iter(["yes", "no", "yes", "yes"])
        cases = [Case("a", {"x": "1"}, [{"type": "contains", "value": "yes"}])]
        r = run_prompt(v, cases, lambda p: next(seq), samples=4, threshold=0.75)
        self.assertTrue(r.results[0].passed)

    def test_model_errors_count_as_failed_samples(self):
        v = self._v()
        n = {"i": 0}
        def m(p):
            n["i"] += 1
            if n["i"] == 1:
                raise RuntimeError("boom")
            return "yes"
        cases = [Case("a", {"x": "1"}, [{"type": "contains", "value": "yes"}])]
        r = run_prompt(v, cases, m, samples=2, threshold=0.5)
        self.assertTrue(r.results[0].passed)
        self.assertEqual(r.results[0].pass_count, 1)

    def test_bad_args_and_legacy_load(self):
        v = self._v()
        with self.assertRaises(ValueError):
            run_prompt(v, [], lambda p: "", samples=0)
        with self.assertRaises(ValueError):
            run_prompt(v, [], lambda p: "", threshold=0)
        old = CaseResult("z", True, 1.0, "ok")
        self.assertEqual((old.samples, old.pass_count), (1, 1))
        self.assertEqual(CaseResult("z", False, 0.0, "").pass_count, 0)

    def test_cli_samples_flag(self):
        with tempfile.TemporaryDirectory() as d:
            pf, cf = Path(d, "p.txt"), Path(d, "c.jsonl")
            pf.write_text("Say {{x}}")
            cf.write_text(json.dumps({"id": "a", "vars": {"x": "hi"}, "checks": [{"type": "min_words", "value": 1}]}) + "\n")
            main(["--root", d + "/reg", "commit", "p", "--file", str(pf)])
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = main(["--root", d + "/reg", "run", "p", "--cases", str(cf), "--samples", "3", "-v"])
            self.assertEqual(rc, 0)
            self.assertIn("[3/3]", buf.getvalue())


    def test_check_summary_file(self):
        with tempfile.TemporaryDirectory() as d:
            pf, cf = Path(d, "p.txt"), Path(d, "c.jsonl")
            pf.write_text("Say {{x}}")
            cf.write_text(json.dumps({"id": "a", "vars": {"x": "hi"}, "checks": [{"type": "contains", "value": "hi"}]}) + "\n")
            main(["--root", d + "/reg", "commit", "p", "--file", str(pf)])
            main(["--root", d + "/reg", "baseline", "p", "--cases", str(cf), "--out", d + "/b.json"])
            sm = Path(d, "sum.md")
            with redirect_stdout(io.StringIO()):
                rc = main(["--root", d + "/reg", "check", "p", "--cases", str(cf), "--baseline", d + "/b.json", "--summary", str(sm)])
            self.assertEqual(rc, 0)
            self.assertIn("PASS", sm.read_text())

    def test_render_check_markdown_fail(self):
        from promptlab import RegressionReport
        from promptlab.compare import render_check_markdown
        rep = RegressionReport(["t1"], ["t2"], [], [], 0.5, 0.75, False, ["1 regression(s) > allowed 0: t1"])
        md = render_check_markdown(rep)
        self.assertIn("FAIL", md); self.assertIn("`t1`", md); self.assertIn("Fixed", md)


class TemplateTests(unittest.TestCase):
    def test_variables_and_required(self):
        t = Template("Hi {{ name }}, {{greeting|hello}} {{name}} {{x}}")
        self.assertEqual(t.variables(), ["name", "greeting", "x"])
        self.assertEqual(t.required(), ["name", "x"])

    def test_render_with_default(self):
        self.assertEqual(Template("{{a}}-{{b|dflt}}").render({"a": 1}), "1-dflt")
        self.assertEqual(Template("{{a|x}}").render({"a": "y"}), "y")

    def test_strict_missing_and_unknown(self):
        with self.assertRaises(TemplateError):
            Template("{{a}}").render({})
        with self.assertRaises(TemplateError):
            Template("{{a}}").render({"a": 1, "typo": 2})
        self.assertEqual(Template("{{a}}").render({}, strict=False), "{{a}}")

    def test_json_braces_are_not_placeholders(self):
        t = Template('Reply {"category": "x"} to {{q}}')
        self.assertEqual(t.variables(), ["q"])
        self.assertEqual(t.render({"q": "me"}), 'Reply {"category": "x"} to me')

    def test_digest_ignores_trailing_whitespace(self):
        self.assertEqual(Template("a  \nb\n\n").digest(), Template("a\nb").digest())
        self.assertNotEqual(Template("a").digest(), Template("b").digest())


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = Registry(self.tmp.name, clock=lambda: 0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_commit_numbers_and_noop(self):
        v1, c1 = self.reg.commit("p", "one {{x}}", "first")
        v1b, c1b = self.reg.commit("p", "one {{x}}  \n")
        v2, c2 = self.reg.commit("p", "two {{x}}")
        self.assertEqual((v1.number, c1, c1b, v2.number, c2), (1, True, False, 2, True))
        self.assertEqual(v1b.number, 1)
        self.assertEqual(v2.parent, 1)
        self.assertEqual(v1.created, "1970-01-01T00:00:00Z")

    def test_resolve_refs(self):
        self.reg.commit("p", "a")
        v2, _ = self.reg.commit("p", "b")
        self.reg.tag("p", "prod", "v1")
        self.assertEqual(self.reg.resolve("p").number, 2)
        self.assertEqual(self.reg.resolve("p", "1").number, 1)
        self.assertEqual(self.reg.resolve("p", "v2").number, 2)
        self.assertEqual(self.reg.resolve("p", "prod").number, 1)
        self.assertEqual(self.reg.resolve("p", v2.hash[:8]).number, 2)
        for bad in ("v9", "nope", "zzzz"):
            with self.assertRaises(RegistryError):
                self.reg.resolve("p", bad)

    def test_tag_moves_and_rejects_reserved(self):
        self.reg.commit("p", "a"); self.reg.commit("p", "b")
        self.reg.tag("p", "prod", "v1"); self.reg.tag("p", "prod", "v2")
        self.assertEqual(self.reg.resolve("p", "prod").number, 2)
        for bad in ("latest", "v3", "7", "bad tag"):
            with self.assertRaises(RegistryError):
                self.reg.tag("p", bad)

    def test_invalid_name_empty_text_and_missing(self):
        with self.assertRaises(RegistryError):
            self.reg.commit("../evil", "x")
        with self.assertRaises(RegistryError):
            self.reg.commit("p", "  \n")
        with self.assertRaises(RegistryError):
            self.reg.resolve("ghost")

    def test_diff_and_names(self):
        self.reg.commit("p", "line1\nline2"); self.reg.commit("p", "line1\nline3")
        d = self.reg.diff("p", "v1", "v2")
        self.assertIn("-line2", d); self.assertIn("+line3", d)
        self.assertEqual(self.reg.names(), ["p"])


class CheckTests(unittest.TestCase):
    def test_basic_checks(self):
        self.assertTrue(run_check({"type": "contains", "value": "REFUND"}, "a refund").passed)
        self.assertFalse(run_check({"type": "contains", "value": "x", "ignore_case": False}, "X").passed)
        self.assertTrue(run_check({"type": "not_contains", "value": "sorry"}, "ok").passed)
        self.assertTrue(run_check({"type": "equals", "value": "yes"}, " yes\n").passed)
        self.assertTrue(run_check({"type": "regex", "value": r"\d{3}"}, "ab 123").passed)
        self.assertTrue(run_check({"type": "max_words", "value": 3}, "a b c").passed)
        self.assertFalse(run_check({"type": "max_words", "value": 2}, "a b c").passed)
        self.assertTrue(run_check({"type": "min_words", "value": 2}, "a b").passed)
        self.assertFalse(run_check({"type": "max_chars", "value": 2}, "abc").passed)

    def test_json_checks(self):
        out = '```json\n{"a": {"b": [1, 2]}, "c": "x"}\n```'
        self.assertTrue(run_check({"type": "json_valid"}, out).passed)
        self.assertTrue(run_check({"type": "json_field", "path": "a.b.1", "value": 2}, out).passed)
        self.assertFalse(run_check({"type": "json_field", "path": "c", "value": "y"}, out).passed)
        self.assertFalse(run_check({"type": "json_field", "path": "zzz"}, out).passed)
        self.assertTrue(run_check({"type": "json_field", "path": "c"}, out).passed)
        self.assertFalse(run_check({"type": "json_valid"}, "not json").passed)
        self.assertFalse(run_check({"type": "json_field", "path": "a"}, "nope").passed)

    def test_bad_checks_fail_not_raise(self):
        self.assertFalse(run_check({"type": "regex", "value": "("}, "x").passed)
        self.assertFalse(run_check({"type": "wat"}, "x").passed)
        self.assertFalse(run_check({"type": "contains"}, "x").passed)


class StatsTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(0, 10)
        self.assertEqual(lo, 0.0); self.assertGreater(hi, 0.2)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual((lo + hi) / 2, 0.5, places=2)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))
        self.assertLessEqual(wilson(10, 10)[1], 1.0)

    def test_mcnemar_known_values(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 10), 2 / 1024)
        self.assertAlmostEqual(mcnemar_exact(1, 9), 2 * 11 / 1024)
        self.assertEqual(mcnemar_exact(3, 8), mcnemar_exact(8, 3))

    def test_bootstrap(self):
        a = [0.0] * 30
        b = [1.0] * 30
        m, lo, hi = paired_bootstrap(a, b)
        self.assertEqual((m, lo, hi), (1.0, 1.0, 1.0))
        a = [1, 0, 1, 0] * 10
        m, lo, hi = paired_bootstrap(a, a)
        self.assertEqual(m, 0.0); self.assertLessEqual(lo, 0.0 + 1e-12); self.assertGreaterEqual(hi, 0.0)
        self.assertEqual(paired_bootstrap(a, a, seed=3), paired_bootstrap(a, a, seed=3))
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 2])


def mk_run(version, passes, prompt="p"):
    return RunResult(prompt, version, f"h{version}",
                     [CaseResult(f"c{i}", ok, 1.0 if ok else 0.0, "", []) for i, ok in enumerate(passes)])


class CompareTests(unittest.TestCase):
    def test_better_verdict(self):
        a = mk_run(1, [False] * 12 + [True] * 8)
        b = mk_run(2, [True] * 20)
        c = compare_runs(a, b)
        self.assertEqual(len(c.only_b), 12); self.assertEqual(c.only_a, [])
        self.assertIn("B is better", c.verdict)
        self.assertGreater(c.score_ci[0], 0)
        self.assertIn("Verdict: B is better", render_markdown(c))

    def test_worse_and_identical_and_underpowered(self):
        a = mk_run(1, [True] * 20); b = mk_run(2, [False] * 10 + [True] * 10)
        self.assertIn("A is better", compare_runs(a, b).verdict)
        self.assertIn("identical", compare_runs(a, mk_run(3, [True] * 20)).verdict)
        small = compare_runs(mk_run(1, [True, False, True, True]), mk_run(2, [True, True, True, True]))
        self.assertIn("no significant", small.verdict)
        self.assertTrue(small.notes)

    def test_mismatched_cases_ignored_and_reported(self):
        a = mk_run(1, [True] * 5); b = mk_run(2, [True] * 3)
        c = compare_runs(a, b)
        self.assertEqual(c.n, 3); self.assertTrue(any("ignored" in n for n in c.notes))
        with self.assertRaises(ValueError):
            compare_runs(a, RunResult("p", 2, "h", [CaseResult("zz", True, 1.0, "", [])]))

    def test_regression_gate(self):
        base = mk_run(1, [True, True, False, True])
        cur = mk_run(2, [True, False, True, True])
        r = check_regression(cur, base)
        self.assertFalse(r.ok); self.assertEqual(r.regressions, ["c1"]); self.assertEqual(r.fixed, ["c2"])
        self.assertTrue(check_regression(cur, base, max_regressions=1).ok)
        self.assertFalse(check_regression(cur, base, max_regressions=1, min_pass_rate=0.9).ok)
        self.assertFalse(check_regression(mk_run(2, [False] * 4), base, max_regressions=9, max_pass_rate_drop=0.1).ok)

    def test_regression_missing_and_new_cases(self):
        base = mk_run(1, [True] * 4); cur = mk_run(2, [True] * 3 + [True, True])
        r = check_regression(cur, base)
        self.assertTrue(r.ok); self.assertEqual(r.new_cases, ["c4"])
        r = check_regression(mk_run(2, [True] * 2), base)
        self.assertFalse(r.ok); self.assertEqual(r.missing_cases, ["c2", "c3"])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = Registry(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_scores_partial_and_errors(self):
        v, _ = self.reg.commit("p", "Say {{word}}")
        cases = [
            Case("a", {"word": "hi"}, [{"type": "contains", "value": "hi"}, {"type": "max_words", "value": 1}]),
            Case("b", {}, [{"type": "contains", "value": "x"}]),          # render error
            Case("c", {"word": "boom"}, [{"type": "contains", "value": "x"}]),  # model error
        ]
        def model(p):
            if "boom" in p:
                raise RuntimeError("down")
            return "hi there"
        r = run_prompt(v, cases, model)
        by = r.by_id()
        self.assertFalse(by["a"].passed); self.assertEqual(by["a"].score, 0.5)
        self.assertIn("max_words", by["a"].failures[0])
        self.assertIn("render", by["b"].error); self.assertIn("model: down", by["c"].error)
        self.assertEqual(r.passed, 0)

    def test_load_cases_validation(self):
        p = Path(self.tmp.name) / "c.jsonl"
        p.write_text('# c\n{"id":"a","vars":{},"checks":[{"type":"json_valid"}]}\n\n')
        self.assertEqual(len(load_cases(p)), 1)
        for bad in ('{"id":"a","vars":{},"checks":[]}', 'not json', '{"vars":{}}',
                    '{"id":"a","checks":[{"type":"json_valid"}]}\n{"id":"a","checks":[{"type":"json_valid"}]}'):
            p.write_text(bad)
            with self.assertRaises(ValueError):
                load_cases(p)

    def test_runresult_roundtrip(self):
        r = mk_run(1, [True, False])
        p = Path(self.tmp.name) / "r.json"
        r.save(p)
        self.assertEqual(RunResult.load(p).to_dict(), r.to_dict())


class ModelTests(unittest.TestCase):
    def test_rule_model_is_deterministic_and_prompt_sensitive(self):
        m = RuleModel()
        p = "Categories: billing, bug.\nRespond in JSON.\n<ticket>I need a refund for the charge</ticket>"
        self.assertEqual(m(p), m(p))
        self.assertIn("category", json.loads(m(p)))
        self.assertEqual(RuleModel()._classify("app crash and error"), "bug")
        self.assertEqual(RuleModel()._classify("forgot my password"), "account")

    def test_quality_ordering_holds_over_examples(self):
        cases = load_cases(EX / "ticket_cases.jsonl")
        reg = Registry(tempfile.mkdtemp())
        rates = []
        for i in (1, 2, 3):
            v, _ = reg.commit("t", (EX / f"triage_v{i}.txt").read_text())
            rates.append(run_prompt(v, cases, RuleModel()).pass_rate)
        self.assertLess(rates[0], rates[1]); self.assertLess(rates[1], rates[2])
        self.assertGreater(rates[2], 0.9)

    def test_make_model_and_command(self):
        self.assertIsInstance(make_model("rule:3"), RuleModel)
        with self.assertRaises(ValueError):
            make_model("gpt")
        out = make_model(f"cmd:{sys.executable} -c \"import sys; print(sys.stdin.read().upper())\"")("abc")
        self.assertEqual(out.strip(), "ABC")
        with self.assertRaises(RuntimeError):
            CommandModel(f"{sys.executable} -c \"import sys; sys.exit(3)\"")("x")


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_full_workflow(self):
        with tempfile.TemporaryDirectory() as root:
            base = ["--root", root]
            cases = str(EX / "ticket_cases.jsonl")
            for i in (1, 2, 3):
                code, out, _ = self.run_cli(*base, "commit", "triage", "--file", str(EX / f"triage_v{i}.txt"), "-m", f"v{i}")
                self.assertEqual(code, 0); self.assertIn("committed", out)
            code, out, _ = self.run_cli(*base, "commit", "triage", "--file", str(EX / "triage_v3.txt"))
            self.assertIn("unchanged", out)
            self.assertIn("triage", self.run_cli(*base, "list")[1])
            self.assertEqual(self.run_cli(*base, "tag", "triage", "prod", "--ref", "v2")[0], 0)
            self.assertIn("[prod]", self.run_cli(*base, "log", "triage")[1])
            self.assertIn("Categories", self.run_cli(*base, "show", "triage", "--ref", "prod")[1])
            self.assertIn("+Categories", self.run_cli(*base, "diff", "triage", "v1", "v2")[1])
            code, out, _ = self.run_cli(*base, "compare", "triage", "v1", "v3", "--cases", cases,
                                        "--report", str(Path(root) / "r.md"))
            self.assertIn("B is better", out); self.assertTrue((Path(root) / "r.md").exists())
            bl = str(Path(root) / "bl.json")
            self.assertEqual(self.run_cli(*base, "baseline", "triage", "--ref", "v3", "--cases", cases, "--out", bl)[0], 0)
            code, out, _ = self.run_cli(*base, "check", "triage", "--ref", "v3", "--cases", cases, "--baseline", bl)
            self.assertEqual(code, 0); self.assertIn("OK", out)
            code, out, _ = self.run_cli(*base, "check", "triage", "--ref", "v1", "--cases", cases, "--baseline", bl)
            self.assertEqual(code, 1); self.assertIn("FAIL", out)
            code, out, _ = self.run_cli(*base, "check", "triage", "--ref", "v1", "--cases", cases, "--baseline", bl,
                                        "--max-regressions", "100")
            self.assertEqual(code, 0)

    def test_errors_exit_2(self):
        with tempfile.TemporaryDirectory() as root:
            code, _, err = self.run_cli("--root", root, "show", "ghost")
            self.assertEqual(code, 2); self.assertIn("no such prompt", err)


if __name__ == "__main__":
    unittest.main()
