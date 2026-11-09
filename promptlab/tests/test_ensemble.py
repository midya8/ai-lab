import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from promptlab import compare_ensemble, render_ensemble, fleiss_kappa, RulePairJudge, PositionBiasedJudge
from promptlab.ensemble import vote

RUBRIC = 'Says "refund".'


def always(slot):
    return lambda prompt: json.dumps({"winner": slot})


def boom(prompt):
    raise RuntimeError("down")


def items(n=8):
    return [{"id": str(i), "rubric": RUBRIC, "a": "Thanks", "b": "We issue a refund"} for i in range(n)]


class KappaTests(unittest.TestCase):
    def test_perfect_and_chance(self):
        self.assertAlmostEqual(fleiss_kappa([["a", "a"], ["b", "b"], ["tie", "tie"]]), 1.0)
        self.assertAlmostEqual(fleiss_kappa([["a", "b"], ["b", "a"]]), -1.0)

    def test_known_value(self):
        # 2 raters, 4 items, 3 agree: po=.75; marginals a 5/8, b 3/8 -> pe=.53125 -> kappa .4667
        k = fleiss_kappa([["a", "a"], ["b", "b"], ["a", "a"], ["b", "a"]])
        self.assertAlmostEqual(k, (0.75 - 0.53125) / (1 - 0.53125))

    def test_undefined_and_errors(self):
        self.assertIsNone(fleiss_kappa([["a", "a"], ["a", "a"]]))
        with self.assertRaises(ValueError):
            fleiss_kappa([["a"]])
        with self.assertRaises(ValueError):
            fleiss_kappa([])


class VoteTests(unittest.TestCase):
    def test_vote(self):
        self.assertEqual(vote({"x": "a", "y": "a", "z": "b"}, 2), ("a", "majority"))
        self.assertEqual(vote({"x": "a", "y": "b"}, 2), ("tie", "split"))
        self.assertEqual(vote({"x": "a", "y": "error", "z": "error"}, 2), ("tie", "error"))
        self.assertEqual(vote({"x": "b", "y": "error", "z": "b"}, 2), ("b", "majority"))


class EnsembleTests(unittest.TestCase):
    def test_needs_two(self):
        with self.assertRaises(ValueError):
            compare_ensemble({"x": RulePairJudge()}, items())

    def test_unanimous_rule_judges(self):
        r = compare_ensemble({"r1": RulePairJudge(), "r2": RulePairJudge(), "r3": RulePairJudge()}, items())
        self.assertEqual(r.summary.wins_b, 8)
        self.assertEqual(r.unanimity, 1.0)
        self.assertEqual(r.agreement, 1.0)
        self.assertTrue(all(v.unanimous for v in r.verdicts))
        self.assertIn("B is better", r.summary.verdict)

    def test_majority_overrules_one_biased_judge(self):
        biased = PositionBiasedJudge(RulePairJudge(), 1.0, "A")
        r = compare_ensemble({"r1": RulePairJudge(), "r2": RulePairJudge(), "bias": biased}, items())
        self.assertEqual(r.summary.wins_b, 8)
        self.assertEqual(r.per_judge["bias"].wins_b, 0)  # alone it can only flip/tie
        self.assertLess(r.unanimity, 1.0)

    def test_split_becomes_tie(self):
        r = compare_ensemble({"x": always("A"), "y": always("B")}, items(4))
        # always("A") picks slot A in both orders -> position flip -> tie; same for B
        self.assertEqual(r.summary.wins_a + r.summary.wins_b, 0)

    def test_failed_judge_tolerated_within_quorum(self):
        r = compare_ensemble({"r1": RulePairJudge(), "r2": RulePairJudge(), "dead": boom}, items(6))
        self.assertEqual(r.summary.wins_b, 6)
        self.assertEqual(r.full_items, 0)
        self.assertEqual(r.per_judge["dead"].errors, 6)

    def test_below_quorum_is_error(self):
        r = compare_ensemble({"r1": RulePairJudge(), "d1": boom, "d2": boom}, items(3))
        self.assertEqual(r.summary.errors, 3)

    def test_skips_empty_outputs(self):
        its = items(3) + [{"id": "x", "rubric": RUBRIC, "a": "", "b": "y"}]
        r = compare_ensemble({"r1": RulePairJudge(), "r2": RulePairJudge()}, its)
        self.assertEqual(r.skipped, 1)

    def test_render_warns_on_low_kappa(self):
        mixed = [{"id": str(i), "rubric": RUBRIC, "a": "Thanks", "b": "We issue a refund"} for i in range(6)]
        r = compare_ensemble({"r1": RulePairJudge(), "bias": PositionBiasedJudge(RulePairJudge(), 1.0, "A")}, mixed)
        txt = render_ensemble(r, "v1", "v2")
        self.assertIn("Ensemble of 2 judges", txt)
        self.assertIn("Fleiss kappa", txt)


if __name__ == "__main__":
    unittest.main()


class EnsembleGateTests(unittest.TestCase):
    def _res(self, slots):
        from promptlab import ensemble_gate  # noqa: F401
        judges = {f"j{i}": always(s) for i, s in enumerate(slots)}
        return compare_ensemble(judges, items(8))

    def test_candidate_better_passes(self):
        from promptlab import ensemble_gate
        r = self._res(["b", "b", "b"])
        self.assertEqual(ensemble_gate(r, 0.3, 0.0 if r.kappa is not None else None), [])

    def test_candidate_worse_fails_majority(self):
        from promptlab import ensemble_gate
        swapped = [{**it, "a": it["b"], "b": it["a"]} for it in items(8)]  # baseline A now has "refund"
        r = compare_ensemble({"x": RulePairJudge(), "y": RulePairJudge(), "z": RulePairJudge()}, swapped)
        self.assertTrue(any("worse" in m or "baseline wins" in m for m in ensemble_gate(r, 0.3)))

    def test_low_kappa_and_undefined_kappa(self):
        from promptlab import ensemble_gate
        r = self._res(["b", "b"])  # no variation -> kappa undefined
        self.assertIsNone(r.kappa)
        self.assertTrue(any("undefined" in m for m in ensemble_gate(r, None, 0.4)))
        self.assertEqual(ensemble_gate(r, None, None), [])


class ConcurrentJudgeTests(unittest.TestCase):
    def test_workers_match_serial(self):
        judges = {"x": always("a"), "y": always("b"), "z": always("a")}
        its = items(6)
        s = compare_ensemble(judges, its)
        c = compare_ensemble(judges, its, workers=3)
        self.assertEqual([v.to_dict() for v in s.verdicts], [v.to_dict() for v in c.verdicts])
        self.assertEqual(s.kappa, c.kappa)

    def test_judges_really_overlap(self):
        import threading, time
        barrier = threading.Barrier(3, timeout=5)

        def waiting(prompt):
            barrier.wait()          # deadlocks (BrokenBarrierError) unless all 3 run at once
            return json.dumps({"winner": "a"})
        judges = {"x": waiting, "y": waiting, "z": waiting}
        t = time.time()
        r = compare_ensemble(judges, items(1), workers=3)
        self.assertEqual(r.verdicts[0].kind, "majority")
        self.assertLess(time.time() - t, 5)

    def test_bad_workers(self):
        with self.assertRaises(ValueError):
            compare_ensemble({"x": always("a"), "y": always("b")}, items(1), workers=0)
