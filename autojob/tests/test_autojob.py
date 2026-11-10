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
from autojob.analytics import (analytics_json, analytics_report, by_company,
                                compute_analytics, days_in_stage, funnel,
                                history_of, reached_stages, score_stats,
                                stale_applications)
from autojob.followup import (draft_followup, due_followups, followups_sent,
                               last_activity)
from autojob.letter import draft_letter, review_notes
from autojob.match import evidence_bullets, match_profile
from autojob.posting import find_skills, parse_posting, parse_salary
from autojob.prep import build_prep, prep_json, render_prep_markdown
from autojob.profile import ProfileError, load_profile, total_experience_years, validate_profile
from autojob.tailor import build_tailor, render_tailor_markdown, tailor_json
from autojob.tracker import (TrackerError, add_application, read_log,
                             record_followup, set_status, slug)

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
            self.assertTrue((packet / "interview-prep.md").exists())
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
        self.assertEqual(__version__, "0.5.0")


def _fixture_log(tmp):
    """Four applications with known dates and scores.

    A: Acme Analyst, 0.80 — applied 11-01, screening 11-05, interview 11-12,
       offer 11-18, hired 11-20.
    B: Beta Analyst, 0.70 — applied 11-02, screening 11-06, rejected 11-09.
    C: Acme Engineer, 0.40 — applied 11-03, rejected 11-04.
    D: Delta Analyst, 0.60 — applied 11-10, still waiting.
    """
    log = Path(tmp) / "apps.jsonl"
    add_application(log, "Acme", "Analyst", match_score=0.80, today="2026-11-01")
    set_status(log, "APP-001", "screening", today="2026-11-05")
    set_status(log, "APP-001", "interview", today="2026-11-12")
    set_status(log, "APP-001", "offer", today="2026-11-18")
    set_status(log, "APP-001", "hired", today="2026-11-20")
    add_application(log, "Beta", "Analyst", match_score=0.70, today="2026-11-02")
    set_status(log, "APP-002", "screening", today="2026-11-06")
    set_status(log, "APP-002", "rejected", today="2026-11-09")
    add_application(log, "Acme", "Engineer", match_score=0.40, today="2026-11-03")
    set_status(log, "APP-003", "rejected", today="2026-11-04")
    add_application(log, "Delta", "Analyst", match_score=0.60, today="2026-11-10")
    return log


class TestHistory(unittest.TestCase):
    def test_add_seeds_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            entry = add_application(log, "Acme", "Analyst", today="2026-11-08")
            self.assertEqual(entry["history"],
                             [{"status": "applied", "at": "2026-11-08"}])

    def test_set_status_appends_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-01")
            entry = set_status(log, "APP-001", "screening", today="2026-11-05")
            self.assertEqual(entry["history"], [
                {"status": "applied", "at": "2026-11-01"},
                {"status": "screening", "at": "2026-11-05"},
            ])
            # History survives a read round-trip.
            self.assertEqual(read_log(log)[0]["history"], entry["history"])

    def test_same_status_move_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-01")
            entry = set_status(log, "APP-001", "applied", today="2026-11-02")
            self.assertEqual(len(entry["history"]), 1)

    def test_legacy_entry_without_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            log.write_text(json.dumps({
                "id": "APP-001", "company": "Acme", "title": "Analyst",
                "status": "interview", "applied_at": "2026-11-01",
                "match_score": 0.75, "source": None, "packet_dir": None,
            }) + "\n", encoding="utf-8")
            entry = read_log(log)[0]
            # Funnel counts it as having reached interview (current status),
            # but no stay duration is invented.
            self.assertEqual(reached_stages(entry), {"applied", "interview"})
            self.assertEqual(days_in_stage([entry]), {})
            # The first recorded move backfills the current stage at applied_at.
            entry = set_status(log, "APP-001", "offer", today="2026-11-20")
            self.assertEqual(entry["history"], [
                {"status": "applied", "at": "2026-11-01"},
                {"status": "interview", "at": "2026-11-01"},
                {"status": "offer", "at": "2026-11-20"},
            ])


class TestAnalytics(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_funnel_counts_and_conversions(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            result = funnel(entries)
            self.assertEqual(result["total"], 4)
            self.assertEqual(result["reached"], {
                "applied": 4, "screening": 2, "interview": 1,
                "offer": 1, "hired": 1})
            self.assertEqual(result["conversions"]["screening"], 0.5)
            self.assertEqual(result["conversions"]["interview"], 0.5)
            self.assertEqual(result["conversions"]["offer"], 1.0)
            self.assertEqual(result["conversions"]["hired"], 1.0)
            self.assertEqual(result["response_rate"], 0.5)
            self.assertEqual(result["rejection_rate"], 0.5)
            self.assertEqual(result["active"], 1)
            self.assertEqual(result["outcomes"],
                             {"hired": 1, "rejected": 2, "applied": 1})

    def test_funnel_empty_log(self):
        result = funnel([])
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["reached"]["applied"], 0)
        self.assertIsNone(result["conversions"]["screening"])
        self.assertIsNone(result["response_rate"])

    def test_median_days_in_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            result = days_in_stage(entries)
            # applied stays: A 4d, B 4d, C 1d -> median 4; screening: 7d, 3d -> 5
            self.assertEqual(result["applied"], {"median_days": 4.0, "n": 3})
            self.assertEqual(result["screening"], {"median_days": 5.0, "n": 2})
            self.assertEqual(result["interview"], {"median_days": 6.0, "n": 1})
            self.assertEqual(result["offer"], {"median_days": 2.0, "n": 1})

    def test_median_days_ignores_current_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            # D is still in "applied" with no end date: no stay for D.
            result = days_in_stage([e for e in entries if e["id"] == "APP-004"])
            self.assertEqual(result, {})

    def test_stale_detection_and_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            today = datetime.date(2026, 11, 30)
            stale = stale_applications(entries, 14, today)
            # Only D is non-terminal; last moved 11-10, 20 days ago.
            self.assertEqual([s["id"] for s in stale], ["APP-004"])
            self.assertEqual(stale[0]["days_since_last_move"], 20)
            # Exactly stale_days ago is not stale yet.
            entry = {"id": "APP-099", "company": "Edge", "title": "Analyst",
                     "status": "applied", "applied_at": "2026-11-16",
                     "history": [{"status": "applied", "at": "2026-11-16"}]}
            self.assertEqual(stale_applications([entry], 14, today), [])
            entry["history"] = [{"status": "applied", "at": "2026-11-15"}]
            entry["applied_at"] = "2026-11-15"
            self.assertEqual(len(stale_applications([entry], 14, today)), 1)

    def test_by_company(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            rows = by_company(entries)
            self.assertEqual([r["company"] for r in rows],
                             ["Acme", "Beta", "Delta"])
            acme = rows[0]
            self.assertEqual(acme["applications"], 2)
            self.assertEqual(acme["best_stage"], "hired")
            self.assertEqual(acme["rejected"], 1)
            self.assertEqual(acme["hired"], 1)

    def test_score_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            stats = score_stats(entries)
            self.assertEqual(stats["n"], 4)
            self.assertAlmostEqual(stats["average"], 0.625)
            self.assertAlmostEqual(stats["median"], 0.65)
            self.assertEqual((stats["min"], stats["max"]), (0.4, 0.8))
            # Only A reached interview+; only C was rejected at applied.
            self.assertAlmostEqual(stats["average_reached_interview"], 0.8)
            self.assertEqual(stats["n_reached_interview"], 1)
            self.assertAlmostEqual(stats["average_rejected_at_applied"], 0.4)
            self.assertEqual(stats["n_rejected_at_applied"], 1)

    def test_score_stats_without_scores(self):
        entries = [{"id": "APP-001", "company": "Acme", "title": "Analyst",
                    "status": "applied", "applied_at": "2026-11-01",
                    "match_score": None}]
        self.assertIsNone(score_stats(entries))

    def test_full_report_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = read_log(_fixture_log(tmp))
            data = compute_analytics(entries, today=datetime.date(2026, 11, 30),
                                     stale_days=14)
            text = analytics_report(data)
            self.assertIn("Applications: 4", text)
            self.assertIn("screening: 2  (50.0% of previous stage)", text)
            self.assertIn("Response rate (reached screening): 50.0%", text)
            self.assertIn("applied: 4.0 days (n=3)", text)
            self.assertIn("Stale — no movement in 14+ days: 1", text)
            self.assertIn("APP-004", text)
            self.assertIn("Acme: 2 application(s), best stage hired", text)
            parsed = json.loads(analytics_json(data))
            self.assertEqual(parsed["total"], 4)
            self.assertEqual(parsed["stale_days"], 14)

    def test_empty_report_text(self):
        data = compute_analytics([], today=datetime.date(2026, 11, 30))
        self.assertIn("Applications: 0", analytics_report(data))
        self.assertIn("No applications logged yet", analytics_report(data))

    def test_report_cli_json_and_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            code, out = self._run(["report", "--log", str(log), "--json",
                                   "--today", "2026-11-30"])
            self.assertEqual(code, 0)
            parsed = json.loads(out)
            self.assertEqual(parsed["total"], 4)
            self.assertEqual(parsed["reached"]["hired"], 1)
            code, out = self._run(["report", "--log", str(log),
                                   "--today", "2026-11-30", "--stale-days", "30"])
            self.assertEqual(code, 0)
            self.assertIn("Stale — no movement in 30+ days: none", out)

    def test_report_cli_bad_today_and_stale_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            code, _ = self._run(["report", "--log", str(log),
                                 "--today", "November 30"])
            self.assertEqual(code, 1)
            code, _ = self._run(["report", "--log", str(log), "--stale-days", "-1"])
            self.assertEqual(code, 1)

    def test_report_cli_missing_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(["report", "--log", str(Path(tmp) / "nope.jsonl")])
            self.assertEqual(code, 0)
            self.assertIn("Applications: 0", out)


class TestPrep(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _prep(self):
        return build_prep(example_profile(), example_posting())

    def test_talking_points_have_years_and_verbatim_bullets(self):
        prep = self._prep()
        by_skill = {p["skill"]: p for p in prep["talking_points"]}
        sql = by_skill["SQL"]
        self.assertEqual((sql["years"], sql["level"]), (4, "advanced"))
        self.assertEqual(sql["kind"], "required")
        self.assertTrue(any("Wrote SQL queries against the orders warehouse"
                            in b["bullet"] for b in sql["bullets"]))
        power_bi = by_skill["Power BI"]
        self.assertTrue(any("Built a Power BI dashboard" in b["bullet"]
                            for b in power_bi["bullets"]))
        # Every bullet is quoted verbatim from the profile, with attribution.
        source = {b for job in example_profile()["experience"] for b in job["bullets"]}
        for point in prep["talking_points"]:
            for item in point["bullets"]:
                self.assertIn(item["bullet"], source)
                self.assertIn(item["company"], ("Northwind Traders", "Contoso Retail"))

    def test_talking_point_without_bullet_is_flagged_not_invented(self):
        prep = self._prep()
        by_skill = {p["skill"]: p for p in prep["talking_points"]}
        # Stakeholder Management is matched but no bullet mentions it.
        point = by_skill["Stakeholder Management"]
        self.assertEqual(point["bullets"], [])
        self.assertEqual(point["years"], 3)

    def test_gaps_never_appear_as_talking_points(self):
        prep = self._prep()
        talked = {p["skill"] for p in prep["talking_points"]}
        for missing in ("Communication", "dbt", "Snowflake", "A/B Testing"):
            self.assertNotIn(missing, talked)
        gaps = {(g["skill"], g["kind"]) for g in prep["gap_prep"]}
        self.assertIn(("Communication", "required"), gaps)
        self.assertIn(("dbt", "preferred"), gaps)
        self.assertIn(("Snowflake", "preferred"), gaps)

    def test_likely_questions_use_facts_only(self):
        prep = self._prep()
        questions = prep["likely_questions"]
        self.assertTrue(any("SQL" in q for q in questions))
        self.assertTrue(any("Power BI" in q for q in questions))
        self.assertTrue(any("Data Analyst" in q and "Northwind Traders" in q
                            for q in questions))
        self.assertTrue(any("Data Analyst" in q and "Northwind Analytics" in q
                            for q in questions))
        self.assertTrue(any("salary expectations" in q for q in questions))
        # No question names a skill the profile does not have.
        for missing in ("dbt", "Snowflake", "Communication"):
            self.assertFalse(any(missing in q for q in questions))

    def test_story_bank_most_recent_first_verbatim(self):
        prep = self._prep()
        stories = prep["story_bank"]
        self.assertEqual(len(stories), 2)
        self.assertEqual((stories[0]["title"], stories[0]["company"]),
                         ("Data Analyst", "Northwind Traders"))
        self.assertEqual(stories[0]["start"], "2022-03")
        profile = example_profile()
        self.assertEqual(stories[0]["bullets"], profile["experience"][0]["bullets"])
        self.assertEqual(stories[1]["bullets"], profile["experience"][1]["bullets"])

    def test_facts_come_from_profile(self):
        prep = self._prep()
        facts = prep["facts"]
        self.assertEqual(facts["notice_weeks"], 2)
        self.assertEqual((facts["salary_min"], facts["salary_max"]), (95000, 120000))
        self.assertEqual(facts["currency"], "CAD")
        self.assertEqual(facts["remote_preference"], "remote-only")
        self.assertEqual(facts["total_experience_years"],
                         total_experience_years(example_profile()))
        self.assertEqual(prep["role"]["salary_raw"], "$95,000 - $120,000")
        self.assertEqual(prep["match"]["verdict"], "strong")

    def test_render_markdown_sections_and_gap_placement(self):
        text = render_prep_markdown(self._prep())
        for section in ("## The role", "## Talking points", "## Gaps",
                        "## Likely questions", "## Your facts", "## Story bank"):
            self.assertIn(section, text)
        self.assertIn("No bullet in the profile mentions this skill", text)
        # Missing skills appear only in the Gaps section, never as claims.
        before_gaps = text.split("## Gaps")[0]
        for missing in ("dbt", "Snowflake", "Communication"):
            self.assertNotIn(missing, before_gaps)
        self.assertIn("do not claim these", text)

    def test_prep_with_no_skills_in_posting(self):
        profile = example_profile()
        posting = {"title": "Analyst", "company": "Acme",
                   "location": None, "remote": None, "salary": None,
                   "required_skills": [], "preferred_skills": []}
        prep = build_prep(profile, posting)
        self.assertEqual(prep["talking_points"], [])
        self.assertEqual(prep["gap_prep"], [])
        self.assertIsNone(prep["match"]["score"])
        self.assertTrue(any("Northwind Traders" in q for q in prep["likely_questions"]))
        text = render_prep_markdown(prep)
        self.assertIn("No matched skills", text)

    def test_prep_json_round_trips(self):
        parsed = json.loads(prep_json(self._prep()))
        self.assertEqual(parsed["role"]["company"], "Northwind Analytics")
        self.assertEqual(len(parsed["talking_points"]), 5)
        self.assertEqual(len(parsed["gap_prep"]), 4)

    def test_prep_cli_markdown_json_and_out(self):
        code, out = self._run(["prep", "--profile", str(EXAMPLE_PROFILE),
                               "--posting", str(EXAMPLE_JOB)])
        self.assertEqual(code, 0)
        self.assertIn("Interview prep", out)
        self.assertIn("Northwind Analytics", out)
        code, out = self._run(["prep", "--profile", str(EXAMPLE_PROFILE),
                               "--posting", str(EXAMPLE_JOB), "--json"])
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["role"]["title"], "Data Analyst")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prep.md"
            code, out = self._run(["prep", "--profile", str(EXAMPLE_PROFILE),
                                   "--posting", str(EXAMPLE_JOB),
                                   "--out", str(path)])
            self.assertEqual(code, 0)
            self.assertTrue(path.exists())
            self.assertIn("Talking points", path.read_text(encoding="utf-8"))

    def test_prep_cli_bad_profile(self):
        code, _ = self._run(["prep", "--profile", "no-such-profile.json",
                             "--posting", str(EXAMPLE_JOB)])
        self.assertEqual(code, 2)


def _followup_entry(status="applied", applied="2026-11-01", moves=()):
    """A tracker entry built by hand, with explicit history events."""
    history = [{"status": "applied", "at": applied}]
    for to_status, at in moves:
        history.append({"status": to_status, "at": at})
    return {
        "id": "APP-001", "company": "Acme", "title": "Analyst",
        "status": status, "applied_at": applied, "history": history,
        "match_score": 0.8, "source": None, "packet_dir": None,
    }


class TestFollowup(unittest.TestCase):
    def test_applied_wait_is_seven_days_with_boundary(self):
        entry = _followup_entry()
        # Exactly 7 days of waiting is not due yet (same rule as stale).
        self.assertEqual(due_followups([entry], datetime.date(2026, 11, 8)), [])
        due = due_followups([entry], datetime.date(2026, 11, 9))
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["days_waiting"], 8)
        self.assertEqual(due[0]["wait_days"], 7)
        self.assertEqual(due[0]["suggestion"], "send a first follow-up")

    def test_per_stage_waits(self):
        screening = _followup_entry("screening", moves=[("screening", "2026-11-05")])
        interview = _followup_entry("interview", moves=[("screening", "2026-11-05"),
                                                        ("interview", "2026-11-10")])
        offer = _followup_entry("offer", moves=[("screening", "2026-11-05"),
                                                ("interview", "2026-11-08"),
                                                ("offer", "2026-11-10")])
        # Screening: last move 11-05, wait 7 — due from 11-13.
        self.assertEqual(due_followups([screening], datetime.date(2026, 11, 12)), [])
        self.assertEqual(len(due_followups([screening], datetime.date(2026, 11, 13))), 1)
        # Interview: last move 11-10, wait 3 — due from 11-14.
        self.assertEqual(due_followups([interview], datetime.date(2026, 11, 13)), [])
        self.assertEqual(len(due_followups([interview], datetime.date(2026, 11, 14))), 1)
        # Offer: last move 11-10, wait 2 — due from 11-13.
        self.assertEqual(due_followups([offer], datetime.date(2026, 11, 12)), [])
        self.assertEqual(len(due_followups([offer], datetime.date(2026, 11, 13))), 1)

    def test_terminal_applications_never_due(self):
        for status in ("hired", "rejected", "withdrawn"):
            entry = _followup_entry(status, moves=[(status, "2026-11-02")])
            self.assertEqual(due_followups([entry], datetime.date(2026, 12, 1)), [])

    def test_wait_days_override_applies_to_every_stage(self):
        entry = _followup_entry()
        # 4 days waiting: a 4-day wait is not due yet (boundary), a 3-day
        # wait is.
        self.assertEqual(
            due_followups([entry], datetime.date(2026, 11, 5), wait_days=4), [])
        self.assertEqual(
            len(due_followups([entry], datetime.date(2026, 11, 5), wait_days=3)), 1)
        with self.assertRaises(ValueError):
            due_followups([entry], datetime.date(2026, 11, 5), wait_days=-1)

    def test_due_list_is_most_overdue_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            entries = read_log(log)
            # At 2026-11-20 only D (applied 11-10, 10 days) is due — A is
            # hired, B and C rejected.
            due = due_followups(entries, datetime.date(2026, 11, 20))
            self.assertEqual([d["id"] for d in due], ["APP-004"])

    def test_draft_uses_facts_only(self):
        entry = _followup_entry()
        draft = draft_followup(entry, "Jordan Lee")
        self.assertEqual(draft["subject"],
                         "Following up — Analyst application (Acme)")
        self.assertIn("Jordan Lee", draft["body"])
        self.assertIn("Analyst", draft["body"])
        self.assertIn("Acme", draft["body"])
        self.assertIn("2026-11-01", draft["body"])
        # No invented recruiter name: the greeting is generic.
        self.assertIn("Hello Hiring team,", draft["body"])
        self.assertNotIn("Dear ", draft["body"])

    def test_record_followup_resets_the_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-01")
            entry = record_followup(log, "APP-001", today="2026-11-15",
                                    note="emailed hiring team")
            self.assertEqual(entry["history"][-1],
                             {"kind": "followup", "at": "2026-11-15",
                              "note": "emailed hiring team"})
            entries = read_log(log)
            self.assertEqual(followups_sent(entries[0]), 1)
            self.assertEqual(last_activity(entries[0]),
                             datetime.date(2026, 11, 15))
            # 5 days after the follow-up: not due (wait 7). 8 days: due
            # again, now framed as a second nudge.
            self.assertEqual(due_followups(entries, datetime.date(2026, 11, 20)), [])
            due = due_followups(entries, datetime.date(2026, 11, 23))
            self.assertEqual(len(due), 1)
            self.assertEqual(due[0]["suggestion"], "follow up again")
            self.assertEqual(due[0]["followups_sent"], 1)

    def test_record_followup_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            # Terminal application: nobody left to follow up with.
            with self.assertRaises(TrackerError):
                record_followup(log, "APP-001", today="2026-11-21")
            # Same-day duplicate: one nudge per day.
            record_followup(log, "APP-004", today="2026-11-15")
            with self.assertRaises(TrackerError):
                record_followup(log, "APP-004", today="2026-11-15")
            # Unknown id.
            with self.assertRaises(TrackerError):
                record_followup(log, "APP-999", today="2026-11-15")

    def test_record_followup_backfills_legacy_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            log.write_text(json.dumps({
                "id": "APP-001", "company": "Acme", "title": "Analyst",
                "status": "applied", "applied_at": "2026-11-01",
                "match_score": 0.75, "source": None, "packet_dir": None,
            }) + "\n", encoding="utf-8")
            entry = record_followup(log, "APP-001", today="2026-11-10")
            self.assertEqual(entry["history"], [
                {"status": "applied", "at": "2026-11-01"},
                {"kind": "followup", "at": "2026-11-10"},
            ])

    def test_followup_does_not_split_a_stage_stay(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-01")
            record_followup(log, "APP-001", today="2026-11-05")
            set_status(log, "APP-001", "screening", today="2026-11-10")
            entries = read_log(log)
            # One 9-day stay in applied — the mid-stage follow-up is activity,
            # not a stage change.
            self.assertEqual(days_in_stage(entries),
                             {"applied": {"median_days": 9.0, "n": 1}})

    def test_followup_resets_the_stale_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "apps.jsonl"
            add_application(log, "Acme", "Analyst", today="2026-11-01")
            record_followup(log, "APP-001", today="2026-11-15")
            entries = read_log(log)
            # Without the follow-up this would be stale (19 days > 14);
            # with it, only 5 days have passed since the last activity.
            from autojob.analytics import stale_applications
            self.assertEqual(
                stale_applications(entries, 14, datetime.date(2026, 11, 20)), [])


class TestFollowupCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_followup_list_with_and_without_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            code, out = self._run(["followup", "--log", str(log),
                                   "--profile", str(EXAMPLE_PROFILE),
                                   "--today", "2026-11-20"])
            self.assertEqual(code, 0)
            self.assertIn("Follow-ups due: 1", out)
            self.assertIn("APP-004", out)
            self.assertIn("Subject: Following up — Analyst application (Delta)", out)
            self.assertIn("Jordan Lee", out)
            code, out = self._run(["followup", "--log", str(log),
                                   "--today", "2026-11-20"])
            self.assertEqual(code, 0)
            self.assertIn("APP-004", out)
            self.assertNotIn("Subject:", out)
            self.assertIn("Pass --profile", out)

    def test_followup_list_none_due(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            code, out = self._run(["followup", "--log", str(log),
                                   "--today", "2026-11-12"])
            self.assertEqual(code, 0)
            self.assertIn("Follow-ups due: none", out)

    def test_followup_json_includes_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            code, out = self._run(["followup", "--log", str(log),
                                   "--profile", str(EXAMPLE_PROFILE),
                                   "--today", "2026-11-20", "--json"])
            self.assertEqual(code, 0)
            parsed = json.loads(out)
            self.assertEqual(len(parsed), 1)
            self.assertEqual(parsed[0]["id"], "APP-004")
            self.assertEqual(parsed[0]["days_waiting"], 10)
            self.assertIn("Delta", parsed[0]["draft"]["subject"])
            code, out = self._run(["followup", "--log", str(log),
                                   "--today", "2026-11-20", "--json"])
            self.assertEqual(json.loads(out)[0]["draft"], None)

    def test_followup_record_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            code, out = self._run(["followup", "--log", str(log),
                                   "--record", "--id", "APP-004",
                                   "--today", "2026-11-15"])
            self.assertEqual(code, 0)
            self.assertIn("recorded a follow-up for APP-004", out)
            self.assertEqual(followups_sent(read_log(log)[3]), 1)
            # Same-day duplicate is refused (exit 6).
            code, _ = self._run(["followup", "--log", str(log),
                                 "--record", "--id", "APP-004",
                                 "--today", "2026-11-15"])
            self.assertEqual(code, 6)
            # Terminal application is refused (exit 6).
            code, _ = self._run(["followup", "--log", str(log),
                                 "--record", "--id", "APP-001",
                                 "--today", "2026-11-21"])
            self.assertEqual(code, 6)

    def test_followup_cli_usage_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _fixture_log(tmp)
            self.assertEqual(self._run(["followup", "--log", str(log),
                                        "--record"])[0], 1)
            self.assertEqual(self._run(["followup", "--log", str(log),
                                        "--id", "APP-004"])[0], 1)
            self.assertEqual(self._run(["followup", "--log", str(log),
                                        "--today", "not-a-date"])[0], 1)
            self.assertEqual(self._run(["followup", "--log", str(log),
                                        "--wait-days", "-1"])[0], 1)


class TestTailor(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _tailor(self):
        return build_tailor(example_profile(), example_posting())

    def test_matrix_covers_every_posting_skill(self):
        tailor = self._tailor()
        rows = {(r["skill"], r["kind"]) for r in tailor["skills_matrix"]}
        self.assertEqual(len(tailor["skills_matrix"]), 9)
        for skill in ("SQL", "Power BI", "Tableau", "Python",
                      "Stakeholder Management", "Communication"):
            self.assertIn((skill, "required"), rows)
        for skill in ("dbt", "Snowflake", "A/B Testing"):
            self.assertIn((skill, "preferred"), rows)
        by_skill = {r["skill"]: r for r in tailor["skills_matrix"]}
        self.assertTrue(by_skill["SQL"]["have"])
        self.assertFalse(by_skill["Communication"]["have"])
        self.assertFalse(by_skill["dbt"]["have"])
        self.assertEqual(by_skill["SQL"]["years"], 4)
        self.assertEqual(by_skill["SQL"]["level"], "advanced")
        self.assertIsNone(by_skill["Communication"]["years"])
        self.assertEqual(by_skill["Communication"]["bullets"], [])

    def test_matrix_bullets_are_verbatim_from_profile(self):
        tailor = self._tailor()
        by_skill = {r["skill"]: r for r in tailor["skills_matrix"]}
        sql = by_skill["SQL"]
        self.assertTrue(any("Wrote SQL queries against the orders warehouse"
                            in b["bullet"] for b in sql["bullets"]))
        power_bi = by_skill["Power BI"]
        self.assertTrue(any("Built a Power BI dashboard" in b["bullet"]
                            for b in power_bi["bullets"]))
        # Every bullet is quoted verbatim from the profile, with attribution.
        source = {b for job in example_profile()["experience"] for b in job["bullets"]}
        for row in tailor["skills_matrix"]:
            for item in row["bullets"]:
                self.assertIn(item["bullet"], source)
                self.assertIn(item["company"], ("Northwind Traders", "Contoso Retail"))
        # A matched skill with no bullet is flagged by an empty list, not invented.
        self.assertEqual(by_skill["Stakeholder Management"]["bullets"], [])
        self.assertEqual(by_skill["Stakeholder Management"]["years"], 3)

    def test_lead_with_order_is_years_then_kind_then_name(self):
        tailor = self._tailor()
        self.assertEqual(tailor["skills_to_lead_with"],
                         ["SQL", "Power BI", "Python",
                          "Stakeholder Management", "Tableau"])
        # Gaps never lead.
        for missing in ("Communication", "dbt", "Snowflake", "A/B Testing"):
            self.assertNotIn(missing, tailor["skills_to_lead_with"])

    def test_role_ranking_most_evidenced_first(self):
        tailor = self._tailor()
        ranking = tailor["role_ranking"]
        self.assertEqual(len(ranking), 2)
        first, second = ranking
        self.assertEqual((first["title"], first["company"]),
                         ("Data Analyst", "Northwind Traders"))
        self.assertEqual(first["matched_count"], 2)
        self.assertEqual(set(first["matched_skills"]), {"SQL", "Power BI"})
        self.assertTrue(any("Built a Power BI dashboard" in b for b in first["bullets"]))
        self.assertEqual((second["title"], second["company"]),
                         ("Reporting Analyst", "Contoso Retail"))
        self.assertEqual(second["matched_count"], 1)
        self.assertEqual(second["matched_skills"], ["Python"])
        # Roles and bullets are verbatim from the profile.
        profile = example_profile()
        self.assertEqual(first["bullets"][:2], profile["experience"][0]["bullets"][:2])

    def test_role_ranking_tiebreak_is_most_recent_first(self):
        profile = example_profile()
        profile["experience"] = [
            {"title": "Old Role", "company": "Old Co", "start": "2019-01",
             "end": "2020-01", "bullets": ["Wrote SQL queries for the team."]},
            {"title": "New Role", "company": "New Co", "start": "2023-01",
             "end": "present", "bullets": ["Wrote SQL reports for the team."]},
            {"title": "Undated Role", "company": "No Date Co",
             "bullets": ["Wrote SQL scripts for the team."]},
        ]
        tailor = build_tailor(profile, example_posting())
        titles = [r["title"] for r in tailor["role_ranking"]]
        self.assertEqual(titles, ["New Role", "Old Role", "Undated Role"])

    def test_gaps_listed_as_not_to_claim(self):
        tailor = self._tailor()
        self.assertEqual(tailor["not_to_claim"],
                         ["Communication", "dbt", "Snowflake", "A/B Testing"])

    def test_summary_ingredients_use_profile_facts(self):
        tailor = self._tailor()
        ingredients = tailor["summary_ingredients"]
        self.assertEqual(ingredients["headline"], "Data Analyst")
        self.assertEqual([(item["skill"], item["years"])
                          for item in ingredients["lead_skills"]],
                         [("SQL", 4), ("Power BI", 3), ("Python", 3)])
        self.assertEqual(len(tailor["education"]), 1)
        self.assertEqual(tailor["education"][0]["degree"],
                         "B.Comm, Business Analytics")

    def test_render_markdown_sections_and_gap_placement(self):
        text = render_tailor_markdown(self._tailor())
        for section in ("## Skills to lead with", "## Skills matrix",
                        "## Experience", "## Summary ingredients",
                        "## Education", "## Do not claim"):
            self.assertIn(section, text)
        self.assertIn("GAP: do not claim", text)
        self.assertIn("Tailored resume", text)
        # Missing skills never appear in the lead-with section.
        lead = text.split("## Skills matrix")[0]
        for missing in ("Communication", "dbt", "Snowflake", "A/B Testing"):
            self.assertNotIn(missing, lead)
        # The matrix flags the gap explicitly where it lists it.
        self.assertIn("### Communication [required] — GAP: do not claim", text)
        # Roles with no evidenced skill are flagged, not dropped.
        self.assertIn("Evidences 2 posting skill(s)", text)

    def test_tailor_with_no_skills_in_posting(self):
        profile = example_profile()
        posting = {"title": "Analyst", "company": "Acme",
                   "location": None, "remote": None, "salary": None,
                   "required_skills": [], "preferred_skills": []}
        tailor = build_tailor(profile, posting)
        self.assertEqual(tailor["skills_matrix"], [])
        self.assertEqual(tailor["skills_to_lead_with"], [])
        self.assertEqual(tailor["not_to_claim"], [])
        self.assertTrue(all(r["matched_count"] == 0 for r in tailor["role_ranking"]))
        text = render_tailor_markdown(tailor)
        self.assertIn("No matched skills", text)
        self.assertIn("nothing to match against", text)

    def test_skill_in_both_lists_is_one_matrix_row(self):
        profile = example_profile()
        posting = {"title": "Analyst", "company": "Acme",
                   "location": None, "remote": None, "salary": None,
                   "required_skills": ["SQL", "Python"],
                   "preferred_skills": ["SQL", "dbt"]}
        tailor = build_tailor(profile, posting)
        skills = [r["skill"] for r in tailor["skills_matrix"]]
        self.assertEqual(skills, ["SQL", "Python", "dbt"])
        self.assertEqual(
            next(r for r in tailor["skills_matrix"] if r["skill"] == "SQL")["kind"],
            "required")

    def test_tailor_json_round_trips(self):
        parsed = json.loads(tailor_json(self._tailor()))
        self.assertEqual(parsed["role"]["company"], "Northwind Analytics")
        self.assertEqual(len(parsed["skills_matrix"]), 9)
        self.assertEqual(parsed["skills_to_lead_with"][0], "SQL")
        self.assertEqual(parsed["role_ranking"][0]["matched_count"], 2)

    def test_tailor_cli_markdown_json_and_out(self):
        code, out = self._run(["tailor", "--profile", str(EXAMPLE_PROFILE),
                               "--posting", str(EXAMPLE_JOB)])
        self.assertEqual(code, 0)
        self.assertIn("Tailored resume", out)
        self.assertIn("Northwind Analytics", out)
        code, out = self._run(["tailor", "--profile", str(EXAMPLE_PROFILE),
                               "--posting", str(EXAMPLE_JOB), "--json"])
        self.assertEqual(code, 0)
        parsed = json.loads(out)
        self.assertEqual(parsed["role"]["title"], "Data Analyst")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tailored.md"
            code, out = self._run(["tailor", "--profile", str(EXAMPLE_PROFILE),
                                   "--posting", str(EXAMPLE_JOB),
                                   "--out", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            self.assertIn("Skills to lead with", path.read_text(encoding="utf-8"))

    def test_tailor_cli_load_error_exit_2(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not json")
            path = fh.name
        code, _ = self._run(["tailor", "--profile", path,
                             "--posting", str(EXAMPLE_JOB)])
        self.assertEqual(code, 2)

    def test_apply_writes_tailored_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                   "--posting", str(EXAMPLE_JOB),
                                   "--out-dir", tmp])
            self.assertEqual(code, 0)
            packet = Path(tmp) / "northwind-analytics-data-analyst"
            tailored = packet / "tailored-resume.md"
            self.assertTrue(tailored.is_file())
            self.assertIn("Skills to lead with",
                          tailored.read_text(encoding="utf-8"))
            self.assertIn("GAP: do not claim",
                          tailored.read_text(encoding="utf-8"))
            # Dry run lists it without writing.
            code, out = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                   "--posting", str(EXAMPLE_JOB),
                                   "--out-dir", tmp, "--dry-run"])
            self.assertEqual(code, 0)
            self.assertIn("tailored-resume.md", out)


if __name__ == "__main__":
    unittest.main()
