"""Tests for autojob — the job-application assistant that never invents facts."""

from __future__ import annotations

import copy
import datetime
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from autojob import __version__
from autojob.__main__ import main
from autojob.letter import draft_letter, review_notes
from autojob.match import evidence_bullets, match_profile
from autojob.posting import find_skills, parse_posting, parse_salary
from autojob.profile import ProfileError, load_profile, total_experience_years, validate_profile
from autojob.tracker import TrackerError, add_application, read_log, set_status, slug

HERE = Path(__file__).resolve().parent
EXAMPLE_PROFILE = HERE.parent / "examples" / "profile.example.json"
EXAMPLE_JOB = HERE.parent / "examples" / "job.example.md"

TODAY = datetime.date(2026, 11, 8)


def example_profile():
    return json.loads(EXAMPLE_PROFILE.read_text(encoding="utf-8"))


def example_posting():
    text = EXAMPLE_JOB.read_text(encoding="utf-8")
    profile = example_profile()
    return parse_posting(text, [s["name"] for s in profile["skills"]])


class TestProfile(unittest.TestCase):
    def test_example_profile_is_valid(self):
        profile = load_profile(EXAMPLE_PROFILE)
        self.assertEqual(profile["name"], "Jordan Lee")
        self.assertEqual(len(profile["skills"]), 6)

    def test_example_profile_is_fictional_demo_data(self):
        profile = example_profile()
        self.assertIn("example.com", profile["email"])
        self.assertIn("555", profile["phone"])

    def test_missing_name(self):
        profile = example_profile()
        del profile["name"]
        errors = validate_profile(profile)
        self.assertTrue(any(e.startswith("name:") for e in errors))

    def test_bad_email(self):
        profile = example_profile()
        profile["email"] = "not-an-email"
        self.assertTrue(any("email" in e for e in validate_profile(profile)))

    def test_bad_start_date(self):
        profile = example_profile()
        profile["experience"][0]["start"] = "March 2022"
        self.assertTrue(any("start" in e for e in validate_profile(profile)))

    def test_duplicate_skill(self):
        profile = example_profile()
        profile["skills"].append({"name": "sql", "years": 1})
        self.assertTrue(any("duplicate" in e for e in validate_profile(profile)))

    def test_salary_range_inverted(self):
        profile = example_profile()
        profile["preferences"]["salary_min"] = 130000
        profile["preferences"]["salary_max"] = 120000
        self.assertTrue(any("salary_min" in e for e in validate_profile(profile)))

    def test_bad_skill_level(self):
        profile = example_profile()
        profile["skills"][0]["level"] = "wizard"
        self.assertTrue(any("level" in e for e in validate_profile(profile)))

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not json")
            path = fh.name
        with self.assertRaises(ProfileError):
            load_profile(path)

    def test_empty_skills_rejected(self):
        profile = example_profile()
        profile["skills"] = []
        self.assertTrue(any("skills" in e for e in validate_profile(profile)))

    def test_total_experience_years(self):
        profile = example_profile()
        # 2020-06..2022-02 (21 months) + 2022-03..2026-11 (57 months) = 6.5 years
        self.assertEqual(total_experience_years(profile, today=(2026, 11)), 6.5)

    def test_unknown_preference_key(self):
        profile = example_profile()
        profile["preferences"]["telepathy"] = True
        self.assertTrue(any("telepathy" in e for e in validate_profile(profile)))


class TestSalary(unittest.TestCase):
    def test_range_with_commas(self):
        result = parse_salary("Salary: $95,000 - $120,000 CAD")
        self.assertEqual((result["min"], result["max"]), (95000, 120000))

    def test_k_format(self):
        result = parse_salary("Pay is $95k-$120k per year")
        self.assertEqual((result["min"], result["max"]), (95000, 120000))

    def test_single_value(self):
        result = parse_salary("Salary $110,000")
        self.assertEqual((result["min"], result["max"]), (110000, 110000))

    def test_no_salary(self):
        self.assertIsNone(parse_salary("No pay mentioned here at all."))


class TestPosting(unittest.TestCase):
    def test_example_posting_parsed(self):
        posting = example_posting()
        self.assertEqual(posting["title"], "Data Analyst")
        self.assertEqual(posting["company"], "Northwind Analytics")
        self.assertEqual(posting["location"], "Toronto, ON (fully remote)")
        self.assertEqual(posting["remote"], "remote")
        self.assertEqual(posting["salary"]["min"], 95000)
        self.assertEqual(posting["salary"]["max"], 120000)

    def test_required_vs_preferred(self):
        posting = example_posting()
        self.assertIn("SQL", posting["required_skills"])
        self.assertIn("Power BI", posting["required_skills"])
        self.assertIn("dbt", posting["preferred_skills"])
        self.assertIn("Snowflake", posting["preferred_skills"])
        self.assertNotIn("dbt", posting["required_skills"])

    def test_title_at_split(self):
        posting = parse_posting("# BI Developer at Acme Corp\n\nSQL required.\n")
        self.assertEqual(posting["title"], "BI Developer")
        self.assertEqual(posting["company"], "Acme Corp")

    def test_hybrid_detection(self):
        posting = parse_posting("Title: Analyst\nHybrid role, 2 days in office.\n")
        self.assertEqual(posting["remote"], "hybrid")

    def test_no_remote_signal(self):
        posting = parse_posting("Title: Analyst\nWork from our downtown office.\n")
        self.assertIsNone(posting["remote"])

    def test_sql_word_boundary(self):
        # "MySQL" must not count as the "SQL" skill.
        found = find_skills("We use MySQL every day.", ["SQL", "MySQL"])
        self.assertEqual(found, ["MySQL"])

    def test_single_letter_r(self):
        self.assertEqual(find_skills("R and Python required.", ["R", "Python"]),
                         ["R", "Python"])
        self.assertEqual(find_skills("Reporting to the CFO.", ["R"]), [])

    def test_multiword_skill(self):
        found = find_skills("Power BI dashboards and Power\nBI reports.",
                            ["Power BI"])
        self.assertEqual(found, ["Power BI"])

    def test_profile_skill_not_in_common_list(self):
        posting = parse_posting("Needs Quicksight experience.",
                                profile_skills=["Quicksight"])
        self.assertIn("Quicksight", posting["required_skills"])


class TestMatch(unittest.TestCase):
    def test_example_match_numbers(self):
        profile = example_profile()
        posting = example_posting()
        match = match_profile(profile, posting)
        # Required: SQL, Power BI, Tableau, Python, Stakeholder Management,
        # Communication — profile has 5 of 6 (Communication missing).
        self.assertEqual(match["required_coverage"], round(5 / 6, 3))
        self.assertEqual(match["required_missing"], ["Communication"])
        self.assertEqual(match["preferred_coverage"], 0.0)
        self.assertEqual(match["preferred_missing"], ["dbt", "Snowflake", "A/B Testing"])
        self.assertEqual(match["experience_fit"], round(5 / 6, 3))
        self.assertEqual(match["verdict"], "strong")
        # score = (5/6*0.6 + 0*0.25 + 5/6*0.15) / 1.0
        self.assertAlmostEqual(match["score"], round((5 / 6) * 0.75, 3))

    def test_weak_verdict(self):
        profile = example_profile()
        posting = {"required_skills": ["Rust", "Go", "Kubernetes", "Terraform"],
                   "preferred_skills": []}
        match = match_profile(profile, posting)
        self.assertEqual(match["verdict"], "weak")
        self.assertEqual(match["score"], 0.0)

    def test_no_skills_in_posting(self):
        match = match_profile(example_profile(),
                              {"required_skills": [], "preferred_skills": []})
        self.assertIsNone(match["score"])
        self.assertEqual(match["verdict"], "unknown")

    def test_evidence_bullets_verbatim(self):
        profile = example_profile()
        bullets = evidence_bullets(profile, ["SQL", "Power BI"])
        self.assertTrue(bullets)
        source = {b for job in profile["experience"] for b in job["bullets"]}
        for item in bullets:
            self.assertIn(item["bullet"], source)
            self.assertIn(item["company"], ("Northwind Traders", "Contoso Retail"))

    def test_no_evidence_bullets(self):
        profile = example_profile()
        self.assertEqual(evidence_bullets(profile, ["Rust"]), [])


class TestLetter(unittest.TestCase):
    def test_letter_contains_facts_only(self):
        profile = example_profile()
        posting = example_posting()
        match = match_profile(profile, posting)
        letter = draft_letter(profile, posting, match, today=TODAY)
        self.assertIn("Data Analyst", letter)
        self.assertIn("Northwind Analytics", letter)
        self.assertIn("Jordan Lee", letter)
        self.assertIn("SQL", letter)
        # Skills the profile does NOT have must never appear in the letter.
        for missing in ("dbt", "Snowflake", "Rust", "Communication"):
            self.assertNotIn(missing, letter)

    def test_letter_quotes_verbatim_bullet(self):
        profile = example_profile()
        posting = example_posting()
        match = match_profile(profile, posting)
        letter = draft_letter(profile, posting, match, today=TODAY)
        self.assertIn("Built a Power BI dashboard tracking order fulfilment",
                      letter)

    def test_letter_notes_remote_fit(self):
        profile = example_profile()
        posting = example_posting()
        match = match_profile(profile, posting)
        letter = draft_letter(profile, posting, match, today=TODAY)
        self.assertIn("fully remote", letter)

    def test_review_notes_list_gaps(self):
        profile = example_profile()
        posting = example_posting()
        match = match_profile(profile, posting)
        notes = review_notes(profile, posting, match)
        self.assertIn("Do not claim", notes)
        self.assertIn("Communication", notes)
        self.assertIn("dbt", notes)
        self.assertIn("Before submitting", notes)
        self.assertIn("autojob does not submit", notes)


class TestTracker(unittest.TestCase):
    def test_add_and_dedupe(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            entry = add_application(log, "Acme", "Analyst", today="2026-11-08")
            self.assertEqual(entry["id"], "APP-001")
            self.assertEqual(entry["status"], "applied")
            with self.assertRaises(TrackerError):
                add_application(log, "acme", "analyst", today="2026-11-08")
            self.assertEqual(len(read_log(log)), 1)

    def test_status_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-08")
            entry = set_status(log, "APP-001", "interview")
            self.assertEqual(entry["status"], "interview")
            entry = set_status(log, "APP-001", "offer")
            self.assertEqual(entry["status"], "offer")
            entry = set_status(log, "APP-001", "hired")
            self.assertEqual(entry["status"], "hired")

    def test_illegal_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-08")
            with self.assertRaises(TrackerError):
                set_status(log, "APP-001", "offer")  # applied -> offer skips steps
            set_status(log, "APP-001", "rejected")
            with self.assertRaises(TrackerError):
                set_status(log, "APP-001", "interview")  # terminal state

    def test_unknown_id_and_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-08")
            with self.assertRaises(TrackerError):
                set_status(log, "APP-999", "interview")
            with self.assertRaises(TrackerError):
                set_status(log, "APP-001", "teleported")

    def test_slug(self):
        self.assertEqual(slug("Northwind Analytics"), "northwind-analytics")
        self.assertEqual(slug("Data Analyst (Remote)"), "data-analyst-remote")
        self.assertEqual(slug(""), "unknown")

    def test_empty_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_log(Path(tmp) / "missing.jsonl"), [])


class TestCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_validate_ok(self):
        code, out = self._run(["validate", "--profile", str(EXAMPLE_PROFILE)])
        self.assertEqual(code, 0)
        self.assertIn("profile OK", out)

    def test_validate_bad_profile(self):
        profile = example_profile()
        profile["email"] = "bad"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(profile, fh)
            path = fh.name
        code, _ = self._run(["validate", "--profile", path])
        self.assertEqual(code, 2)

    def test_match_fail_under(self):
        code, _ = self._run(["match", "--profile", str(EXAMPLE_PROFILE),
                             "--posting", str(EXAMPLE_JOB), "--fail-under", "0.99"])
        self.assertEqual(code, 3)
        code, out = self._run(["match", "--profile", str(EXAMPLE_PROFILE),
                               "--posting", str(EXAMPLE_JOB), "--fail-under", "0.5"])
        self.assertEqual(code, 0)
        self.assertIn("strong", out)

    def test_draft_refused_below_min_coverage(self):
        code, _ = self._run(["draft", "--profile", str(EXAMPLE_PROFILE),
                             "--posting", str(EXAMPLE_JOB), "--min-coverage", "0.99"])
        self.assertEqual(code, 4)

    def test_draft_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "letter.md"
            code, text = self._run(["draft", "--profile", str(EXAMPLE_PROFILE),
                                    "--posting", str(EXAMPLE_JOB),
                                    "--out", str(out), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertIn("Northwind Analytics", text)
            self.assertFalse(out.exists())

    def test_apply_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "applications"
            code, out = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                   "--posting", str(EXAMPLE_JOB),
                                   "--out-dir", str(out_dir)])
            self.assertEqual(code, 0)
            self.assertIn("APP-001", out)
            packet = out_dir / "northwind-analytics-data-analyst"
            self.assertTrue((packet / "cover-letter.md").exists())
            self.assertTrue((packet / "match.json").exists())
            self.assertTrue((packet / "review-notes.md").exists())
            log = out_dir / "applications.jsonl"
            self.assertTrue(log.exists())

            code, out = self._run(["list", "--log", str(log)])
            self.assertEqual(code, 0)
            self.assertIn("Northwind Analytics", out)

            code, out = self._run(["status", "--log", str(log),
                                   "--id", "APP-001", "--to", "screening"])
            self.assertEqual(code, 0)
            self.assertIn("screening", out)

            code, out = self._run(["report", "--log", str(log)])
            self.assertEqual(code, 0)
            self.assertIn("Applications: 1", out)
            self.assertIn("screening: 1", out)

            # Second apply to the same role is a duplicate.
            code, _ = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                 "--posting", str(EXAMPLE_JOB),
                                 "--out-dir", str(out_dir)])
            self.assertEqual(code, 5)

    def test_apply_dry_run_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "applications"
            code, out = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                   "--posting", str(EXAMPLE_JOB),
                                   "--out-dir", str(out_dir), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(out_dir.exists())

    def test_status_illegal_via_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                       "--posting", str(EXAMPLE_JOB),
                       "--out-dir", str(tmp), "--log", str(log)])
            code, _ = self._run(["status", "--log", str(log),
                                 "--id", "APP-001", "--to", "hired"])
            self.assertEqual(code, 6)

    def test_version(self):
        self.assertTrue(__version__.startswith("0."))


if __name__ == "__main__":
    unittest.main()
