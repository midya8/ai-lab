import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from promptlab import CostMeter, BudgetExceeded, estimate_tokens, parse_price, run_prompt, Case, Template
from promptlab.registry import Version
from promptlab.__main__ import main


class CostTests(unittest.TestCase):
    def test_estimate_and_price(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("abcde"), 2)
        self.assertEqual(parse_price("0.15,0.6"), (0.15, 0.6))
        for bad in ("1", "a,b", "-1,2", "inf,1"):
            with self.assertRaises(ValueError):
                parse_price(bad)

    def test_meter_counts_and_prices(self):
        m = CostMeter(lambda p: "x" * 40, 1_000_000, 2_000_000, count=len)
        m("a" * 10)
        self.assertEqual((m.usage.calls, m.usage.tokens_in, m.usage.tokens_out), (1, 10, 40))
        self.assertAlmostEqual(m.usage.cost, 10 + 80)

    def test_failed_call_still_bills_input(self):
        def boom(p):
            raise RuntimeError("down")
        m = CostMeter(boom, 1e6, 1e6, count=len)
        with self.assertRaises(RuntimeError):
            m("abc")
        self.assertAlmostEqual(m.usage.cost, 3)

    def test_budget_refuses_before_call(self):
        calls = []
        m = CostMeter(lambda p: calls.append(1) or "", 1e6, 0, budget=5, count=len)
        m("abc"); 
        with self.assertRaises(BudgetExceeded):
            m("abc")
        self.assertEqual(len(calls), 1)

    def test_budget_aborts_run_instead_of_failing_cases(self):
        v = Version(name="p", number=1, hash="h", text="say {{w}}", created="", message="", parent=None)
        cases = [Case(f"c{i}", {"w": "hi"}, [{"type": "contains", "value": "hi"}]) for i in range(5)]
        m = CostMeter(lambda p: "hi", 1e6, 0, budget=12, count=len)  # prompt "say hi" = 6 chars
        with self.assertRaises(BudgetExceeded):
            run_prompt(v, cases, m)
        self.assertEqual(m.usage.calls, 2)

    def test_cli_cost_and_budget_exit_code(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "p.txt").write_text("Classify the ticket: <ticket>{{t}}</ticket> Answer in one word.")
            (d / "c.jsonl").write_text("\n".join(
                '{"id":"c%d","vars":{"t":"I was charged twice"},"checks":[{"type":"contains","value":"billing"}]}' % i
                for i in range(4)) + "\n")
            root = str(d / "reg")
            base = ["--root", root]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(base + ["commit", "t", "--file", str(d / "p.txt")]), 0)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = main(base + ["run", "t", "--cases", str(d / "c.jsonl"), "--price", "1,2"])
            self.assertEqual(rc, 0)
            self.assertIn("cost: $", out.getvalue())
            err = io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                rc = main(base + ["run", "t", "--cases", str(d / "c.jsonl"), "--price", "1000000,0", "--budget", "0.00002"])
            self.assertEqual(rc, 3)
            self.assertIn("budget exceeded", err.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(base + ["run", "t", "--cases", str(d / "c.jsonl"), "--budget", "1"]), 2)


if __name__ == "__main__":
    unittest.main()
