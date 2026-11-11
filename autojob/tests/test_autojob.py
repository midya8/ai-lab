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
from autojob.digest import (build_digest, digest_json, gap_leaderboard,
                            render_digest_markdown)
from autojob.followup import (draft_followup, due_followups, followups_sent,
                               last_activity)
from autojob.interview import (InterviewError, collect_interviews,
                               interview_json, interviews_for,
                               render_interview_markdown,
                               validate_interview)
from autojob.letter import draft_letter, review_notes
from autojob.match import evidence_bullets, match_profile
from autojob.plan import (build_plan, evidence_list, plan_json,
                          quick_wins, render_plan_markdown)
from autojob.posting import find_skills, parse_posting, parse_salary
from autojob.prep import build_prep, prep_json, render_prep_markdown
from autojob.profile import ProfileError, load_profile, total_experience_years, validate_profile
from autojob.rank import rank_json, rank_postings, render_rank_markdown
from autojob.salary import (analyze_salaries, render_salary_markdown,
                            salary_json, target_from_profile)
from autojob.offer import (OfferError, compare_offers, first_year_total,
                           latest_offer, offer_json, render_offer_markdown,
                           total_cash, validate_offer)
from autojob.tailor import build_tailor, render_tailor_markdown, tailor_json
from autojob.tracker import (TrackerError, add_application, read_log,
                             record_followup, record_interview, record_offer,
                             set_status, slug)

HERE = Path(__file__).resolve().parent
EXAMPLE_PROFILE = HERE.parent / "examples" / "profile.example.json"
EXAMPLE_JOB = HERE.parent / "examples" / "job.example.md"
EXAMPLE_JOB2 = HERE.parent / "examples" / "job2.example.md"
EXAMPLE_JOB3 = HERE.parent / "examples" / "job3.example.md"
EXAMPLES_DIR = HERE.parent / "examples"

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

    def test_trailing_currency_code(self):
        result = parse_salary("Salary: $95,000 - $120,000 CAD")
        self.assertEqual(result["currency"], "CAD")
        result = parse_salary("Pay is £40,000 - £50,000 GBP per year")
        self.assertEqual(result["currency"], "GBP")
        result = parse_salary("Salary $110,000 CAD")
        self.assertEqual(result["currency"], "CAD")

    def test_bare_symbol_stays_a_symbol(self):
        result = parse_salary("Salary: $95,000 - $120,000")
        self.assertEqual(result["currency"], "$")
        result = parse_salary("Salary $110,000")
        self.assertEqual(result["currency"], "$")


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
        self.assertEqual(__version__, "0.11.0")


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


def _digest_entries():
    """Three log entries: two with v0.6 skill data, one logged before it."""
    northwind = {
        "id": "APP-001", "company": "Northwind Analytics", "title": "Data Analyst",
        "status": "applied", "applied_at": "2026-10-20",
        "history": [{"status": "applied", "at": "2026-10-20"}],
        "match_score": 0.625,
        "required_matched": ["SQL", "Power BI", "Tableau", "Python",
                             "Stakeholder Management"],
        "required_missing": ["Communication"],
        "preferred_matched": [],
        "preferred_missing": ["dbt", "Snowflake", "A/B Testing"],
    }
    harbour = {
        "id": "APP-002", "company": "Harbour Analytics",
        "title": "Business Intelligence Analyst",
        "status": "screening", "applied_at": "2026-11-01",
        "history": [{"status": "applied", "at": "2026-11-01"},
                    {"status": "screening", "at": "2026-11-02"}],
        "match_score": 0.56,
        "required_matched": ["SQL", "Power BI", "Python"],
        "required_missing": ["Communication"],
        "preferred_matched": [],
        "preferred_missing": ["dbt", "Snowflake", "Looker"],
    }
    legacy = {
        "id": "APP-003", "company": "Old Co", "title": "Analyst",
        "status": "applied", "applied_at": "2026-10-01",
        "history": [{"status": "applied", "at": "2026-10-01"}],
        "match_score": 0.5,
    }
    return [northwind, harbour, legacy]


class TestDigest(unittest.TestCase):
    def test_leaderboard_counts_required_and_preferred(self):
        board = gap_leaderboard(_digest_entries())
        by_skill = {row["skill"]: row for row in board}
        comm = by_skill["Communication"]
        self.assertEqual((comm["asked_required"], comm["missing_required"]), (2, 2))
        self.assertEqual((comm["asked_preferred"], comm["missing_preferred"]), (0, 0))
        self.assertEqual(comm["companies"],
                         ["Harbour Analytics", "Northwind Analytics"])
        dbt = by_skill["dbt"]
        self.assertEqual((dbt["asked_preferred"], dbt["missing_preferred"]), (2, 2))
        self.assertEqual((dbt["asked_required"], dbt["missing_required"]), (0, 0))
        self.assertEqual(by_skill["A/B Testing"]["asked"], 1)
        self.assertEqual(by_skill["Looker"]["missing"], 1)

    def test_leaderboard_order_is_most_missed_first(self):
        skills = [row["skill"] for row in gap_leaderboard(_digest_entries())]
        self.assertEqual(skills, ["Communication", "dbt", "Snowflake",
                                  "A/B Testing", "Looker"])

    def test_leaderboard_skips_entries_without_skill_data(self):
        legacy_only = [_digest_entries()[2]]
        self.assertEqual(gap_leaderboard(legacy_only), [])
        # The legacy entry contributes nothing, not a phantom gap.
        board = gap_leaderboard(_digest_entries())
        self.assertNotIn("Analyst", [row["skill"] for row in board])

    def test_leaderboard_matched_skill_is_not_a_gap(self):
        board = gap_leaderboard(_digest_entries())
        by_skill = {row["skill"]: row for row in board}
        # SQL is asked for twice and matched twice: it is not a gap and
        # does not belong on the leaderboard at all.
        self.assertNotIn("SQL", by_skill)
        self.assertNotIn("Power BI", by_skill)

    def test_leaderboard_same_skill_both_kinds(self):
        entries = [
            {"id": "APP-001", "company": "A Co", "title": "Analyst",
             "status": "applied", "applied_at": "2026-11-01",
             "required_matched": [], "required_missing": ["SQL"],
             "preferred_matched": [], "preferred_missing": []},
            {"id": "APP-002", "company": "B Co", "title": "Analyst",
             "status": "applied", "applied_at": "2026-11-02",
             "required_matched": [], "required_missing": [],
             "preferred_matched": [], "preferred_missing": ["SQL"]},
        ]
        board = gap_leaderboard(entries)
        self.assertEqual(len(board), 1)
        row = board[0]
        self.assertEqual((row["asked_required"], row["missing_required"]), (1, 1))
        self.assertEqual((row["asked_preferred"], row["missing_preferred"]), (1, 1))
        self.assertEqual((row["asked"], row["missing"]), (2, 2))
        self.assertEqual(row["companies"], ["A Co", "B Co"])

    def test_build_digest_combines_funnel_stale_due_and_gaps(self):
        digest = build_digest(_digest_entries(), today=TODAY, stale_days=14)
        self.assertEqual(digest["generated_on"], "2026-11-08")
        self.assertEqual(digest["n_applications"], 3)
        self.assertEqual(digest["n_with_skill_data"], 2)
        self.assertEqual(digest["n_without_skill_data"], 1)
        self.assertEqual(digest["funnel"]["total"], 3)
        self.assertEqual(digest["funnel"]["reached"]["screening"], 1)
        self.assertEqual([item["id"] for item in digest["stale"]],
                         ["APP-003", "APP-001"])
        self.assertEqual([item["id"] for item in digest["due_followups"]],
                         ["APP-003", "APP-001"])
        self.assertEqual(digest["gap_leaderboard"][0]["skill"], "Communication")
        self.assertAlmostEqual(digest["scores"]["average"], 0.562, places=3)

    def test_digest_json_round_trip(self):
        digest = build_digest(_digest_entries(), today=TODAY)
        self.assertEqual(json.loads(digest_json(digest)), digest)

    def test_render_empty_log(self):
        text = render_digest_markdown(build_digest([], today=TODAY))
        self.assertIn("no applications logged yet", text)

    def test_render_marks_entries_without_skill_data(self):
        text = render_digest_markdown(
            build_digest(_digest_entries(), today=TODAY))
        self.assertIn("Communication — missing in required in 2 of 2", text)
        self.assertIn("dbt — missing in preferred in 2 of 2", text)
        self.assertIn("1 application(s) logged before v0.6", text)
        self.assertIn("Due a follow-up: 2", text)
        self.assertIn("Stale — no movement in 14+ days: 2", text)

    def test_render_no_gaps_when_everything_matched(self):
        entries = [{
            "id": "APP-001", "company": "A Co", "title": "Analyst",
            "status": "applied", "applied_at": "2026-11-08",
            "required_matched": ["SQL"], "required_missing": [],
            "preferred_matched": ["dbt"], "preferred_missing": [],
        }]
        text = render_digest_markdown(build_digest(entries, today=TODAY))
        self.assertIn("every skill the postings asked for is in the profile", text)


class TestDigestCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _apply_two(self, tmp):
        for job in (EXAMPLE_JOB, EXAMPLE_JOB2):
            code, _ = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                 "--posting", str(job), "--out-dir", tmp])
            self.assertEqual(code, 0)
        return Path(tmp) / "applications.jsonl"

    def test_digest_empty_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(["digest", "--log",
                                   str(Path(tmp) / "missing.jsonl")])
            self.assertEqual(code, 0)
            self.assertIn("no applications logged yet", out)

    def test_digest_json_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._apply_two(tmp)
            future = (datetime.date.today()
                      + datetime.timedelta(days=8)).isoformat()
            code, out = self._run(["digest", "--log", str(log),
                                   "--today", future, "--json"])
            self.assertEqual(code, 0)
            digest = json.loads(out)
            self.assertEqual(digest["n_applications"], 2)
            self.assertEqual(digest["n_with_skill_data"], 2)
            board = digest["gap_leaderboard"]
            self.assertEqual(board[0]["skill"], "Communication")
            self.assertEqual(board[0]["missing_required"], 2)
            self.assertEqual(len(digest["due_followups"]), 2)

    def test_digest_out_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._apply_two(tmp)
            out_path = Path(tmp) / "digest.md"
            code, out = self._run(["digest", "--log", str(log),
                                   "--out", str(out_path)])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            self.assertIn("Application digest", out_path.read_text(encoding="utf-8"))
            self.assertIn("Communication", out_path.read_text(encoding="utf-8"))

    def test_digest_bad_today(self):
        code, _ = self._run(["digest", "--today", "not-a-date"])
        self.assertEqual(code, 1)

    def test_digest_negative_stale_days(self):
        code, _ = self._run(["digest", "--stale-days", "-1"])
        self.assertEqual(code, 1)

    def test_apply_records_posting_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                 "--posting", str(EXAMPLE_JOB),
                                 "--out-dir", tmp])
            self.assertEqual(code, 0)
            entries = read_log(Path(tmp) / "applications.jsonl")
            self.assertEqual(len(entries), 1)
            entry = entries[0]
            self.assertEqual(entry["required_missing"], ["Communication"])
            self.assertIn("SQL", entry["required_matched"])
            self.assertEqual(entry["preferred_missing"],
                             ["dbt", "Snowflake", "A/B Testing"])
            self.assertEqual(entry["preferred_matched"], [])


class TestPlan(unittest.TestCase):
    def test_evidence_finds_skills_with_no_bullet(self):
        profile = load_profile(EXAMPLE_PROFILE)
        rows = evidence_list(profile, _digest_entries())
        by_skill = {row["skill"]: row for row in rows}
        # Stakeholder Management (3 yrs) and Tableau (2 yrs) are in the
        # profile and matched by the Northwind posting, but no bullet
        # mentions either one.
        self.assertEqual(sorted(by_skill), ["Stakeholder Management", "Tableau"])
        stakeholder = by_skill["Stakeholder Management"]
        self.assertEqual((stakeholder["years"], stakeholder["level"]),
                         (3, "intermediate"))
        self.assertEqual((stakeholder["asked_required"],
                          stakeholder["asked_preferred"]), (1, 0))
        self.assertEqual(stakeholder["companies"], ["Northwind Analytics"])
        self.assertEqual(by_skill["Tableau"]["years"], 2)

    def test_evidenced_skills_are_not_listed(self):
        profile = load_profile(EXAMPLE_PROFILE)
        rows = evidence_list(profile, _digest_entries())
        skills = [row["skill"] for row in rows]
        # SQL, Power BI and Python are all mentioned in a bullet.
        self.assertNotIn("SQL", skills)
        self.assertNotIn("Power BI", skills)
        self.assertNotIn("Python", skills)

    def test_evidence_order_is_most_asked_first(self):
        profile = load_profile(EXAMPLE_PROFILE)
        rows = evidence_list(profile, _digest_entries())
        # Both asked once: alphabetical tiebreak.
        self.assertEqual([row["skill"] for row in rows],
                         ["Stakeholder Management", "Tableau"])

    def test_evidence_same_skill_required_and_preferred(self):
        profile = load_profile(EXAMPLE_PROFILE)
        entries = [
            {"id": "APP-001", "company": "A Co", "title": "Analyst",
             "status": "applied", "applied_at": "2026-11-01",
             "required_matched": ["Tableau"], "required_missing": [],
             "preferred_matched": [], "preferred_missing": []},
            {"id": "APP-002", "company": "B Co", "title": "Analyst",
             "status": "applied", "applied_at": "2026-11-02",
             "required_matched": [], "required_missing": [],
             "preferred_matched": ["Tableau"], "preferred_missing": []},
        ]
        rows = evidence_list(profile, entries)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["asked_required"], row["asked_preferred"]), (1, 1))
        self.assertEqual(row["asked"], 2)
        self.assertEqual(row["companies"], ["A Co", "B Co"])

    def test_learn_reuses_leaderboard_order(self):
        profile = load_profile(EXAMPLE_PROFILE)
        plan = build_plan(profile, _digest_entries(), today=TODAY)
        self.assertEqual([row["skill"] for row in plan["learn"]],
                         ["Communication", "dbt", "Snowflake",
                          "A/B Testing", "Looker"])
        self.assertEqual(plan["learn"][0]["missing_required"], 2)

    def test_suggested_role_is_the_most_recent(self):
        profile = load_profile(EXAMPLE_PROFILE)
        plan = build_plan(profile, _digest_entries(), today=TODAY)
        self.assertEqual(plan["suggested_role"],
                         {"title": "Data Analyst",
                          "company": "Northwind Traders",
                          "start": "2022-03"})

    def test_suggested_role_undated_sorts_last(self):
        profile = {"skills": [{"name": "SQL", "years": 4}],
                   "experience": [
                       {"title": "Old Role", "company": "Old Co",
                        "start": "", "bullets": []},
                       {"title": "New Role", "company": "New Co",
                        "start": "2021-06", "bullets": []}]}
        entries = [{"id": "APP-001", "company": "A Co", "title": "Analyst",
                    "status": "applied", "applied_at": "2026-11-01",
                    "required_matched": ["SQL"], "required_missing": [],
                    "preferred_matched": [], "preferred_missing": []}]
        plan = build_plan(profile, entries, today=TODAY)
        self.assertEqual(plan["suggested_role"]["title"], "New Role")

    def test_build_plan_counts_and_quick_wins(self):
        profile = load_profile(EXAMPLE_PROFILE)
        plan = build_plan(profile, _digest_entries(), today=TODAY)
        self.assertEqual(plan["generated_on"], "2026-11-08")
        self.assertEqual(plan["n_applications"], 3)
        self.assertEqual(plan["n_with_skill_data"], 2)
        self.assertEqual(plan["n_without_skill_data"], 1)
        # The three most-missed skills each close 2 applications.
        self.assertEqual(plan["quick_wins"], [
            {"skill": "Communication", "kind": "learn", "closes": 2},
            {"skill": "dbt", "kind": "learn", "closes": 2},
            {"skill": "Snowflake", "kind": "learn", "closes": 2},
        ])

    def test_quick_wins_prefers_evidence_on_a_tie(self):
        wins = quick_wins(
            [{"skill": "Tableau", "asked": 1}],
            [{"skill": "Looker", "missing": 1, "asked": 1}])
        # Same closes: one bullet beats learning a skill.
        self.assertEqual([item["skill"] for item in wins],
                         ["Tableau", "Looker"])
        self.assertEqual(wins[0]["kind"], "evidence")

    def test_plan_skips_entries_without_skill_data(self):
        profile = load_profile(EXAMPLE_PROFILE)
        legacy_only = [_digest_entries()[2]]
        plan = build_plan(profile, legacy_only, today=TODAY)
        self.assertEqual(plan["evidence"], [])
        self.assertEqual(plan["learn"], [])
        self.assertEqual(plan["quick_wins"], [])
        self.assertEqual(plan["n_without_skill_data"], 1)

    def test_plan_json_round_trip(self):
        profile = load_profile(EXAMPLE_PROFILE)
        plan = build_plan(profile, _digest_entries(), today=TODAY)
        self.assertEqual(json.loads(plan_json(plan)), plan)

    def test_render_empty_log(self):
        profile = load_profile(EXAMPLE_PROFILE)
        text = render_plan_markdown(build_plan(profile, [], today=TODAY))
        self.assertIn("no applications logged yet", text)

    def test_render_nothing_to_close(self):
        profile = load_profile(EXAMPLE_PROFILE)
        entries = [{"id": "APP-001", "company": "A Co", "title": "Analyst",
                    "status": "applied", "applied_at": "2026-11-08",
                    "required_matched": ["SQL"], "required_missing": [],
                    "preferred_matched": [], "preferred_missing": []}]
        text = render_plan_markdown(build_plan(profile, entries, today=TODAY))
        self.assertIn("Nothing to close", text)
        self.assertNotIn("Quick wins", text)

    def test_render_marks_entries_without_skill_data(self):
        profile = load_profile(EXAMPLE_PROFILE)
        text = render_plan_markdown(
            build_plan(profile, _digest_entries(), today=TODAY))
        self.assertIn("Stakeholder Management (3 yrs, intermediate)", text)
        self.assertIn("Communication — missing in required in 2 of 2", text)
        self.assertIn("1 application(s) logged before v0.6", text)
        self.assertIn("Suggested place: your most recent role, Data Analyst "
                      "at Northwind Traders", text)
        self.assertIn("1. learn Communication — helps 2 application(s)", text)


class TestPlanCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _apply_two(self, tmp):
        for job in (EXAMPLE_JOB, EXAMPLE_JOB2):
            code, _ = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                 "--posting", str(job), "--out-dir", tmp])
            self.assertEqual(code, 0)
        return Path(tmp) / "applications.jsonl"

    def test_plan_empty_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = self._run(["plan", "--profile",
                                   str(EXAMPLE_PROFILE), "--log",
                                   str(Path(tmp) / "missing.jsonl")])
            self.assertEqual(code, 0)
            self.assertIn("no applications logged yet", out)

    def test_plan_json_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._apply_two(tmp)
            code, out = self._run(["plan", "--profile",
                                   str(EXAMPLE_PROFILE), "--log", str(log),
                                   "--json"])
            self.assertEqual(code, 0)
            plan = json.loads(out)
            self.assertEqual(plan["n_applications"], 2)
            self.assertEqual(plan["n_with_skill_data"], 2)
            self.assertEqual([row["skill"] for row in plan["evidence"]],
                             ["Stakeholder Management", "Tableau"])
            self.assertEqual(plan["learn"][0]["skill"], "Communication")
            self.assertEqual(plan["quick_wins"][0]["skill"], "Communication")

    def test_plan_out_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._apply_two(tmp)
            out_path = Path(tmp) / "plan.md"
            code, out = self._run(["plan", "--profile",
                                   str(EXAMPLE_PROFILE), "--log", str(log),
                                   "--out", str(out_path)])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Gap-closing plan", text)
            self.assertIn("Stakeholder Management", text)

    def test_plan_bad_today(self):
        code, _ = self._run(["plan", "--profile", str(EXAMPLE_PROFILE),
                             "--today", "not-a-date"])
        self.assertEqual(code, 1)

    def test_plan_missing_profile(self):
        code, _ = self._run(["plan", "--profile", "/nonexistent.json"])
        self.assertEqual(code, 2)

    def test_plan_corrupt_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "bad.jsonl"
            log.write_text("{not json\n", encoding="utf-8")
            code, _ = self._run(["plan", "--profile",
                                 str(EXAMPLE_PROFILE), "--log", str(log)])
            self.assertEqual(code, 2)


class TestRank(unittest.TestCase):
    def test_examples_rank_best_first(self):
        profile = example_profile()
        data = rank_postings(profile, EXAMPLES_DIR, include_applied=True)
        self.assertEqual(data["n_files"], 3)
        self.assertEqual(data["n_scored"], 3)
        self.assertEqual(data["n_skipped"], 0)
        titles = [row["title"] for row in data["ranked"]]
        self.assertEqual(titles, ["Data Analyst",
                                  "Business Intelligence Analyst",
                                  "Machine Learning Engineer"])
        self.assertEqual([row["score"] for row in data["ranked"]],
                         [0.625, 0.562, 0.15])

    def test_title_breaks_a_score_tie(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            # Two postings that match identically (SQL only, matched) get
            # the same perfect score; the alphabetically first title wins.
            for name, title in (("b.md", "B Analyst"), ("a.md", "A Analyst")):
                (Path(tmp) / name).write_text(
                    f"# {title} at Fictional Co\n\n## Requirements\n\n- SQL\n",
                    encoding="utf-8")
            data = rank_postings(profile, tmp)
            self.assertEqual([row["title"] for row in data["ranked"]],
                             ["A Analyst", "B Analyst"])
            self.assertEqual(data["ranked"][0]["score"], 1.0)

    def test_unmeasurable_posting_sorts_last(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "plain.md").write_text(
                "# Mystery Role at Fictional Co\n\nNo skills listed at all.\n",
                encoding="utf-8")
            (Path(tmp) / "sql.md").write_text(
                "# Data Analyst at Fictional Co\n\n## Requirements\n\n- SQL\n",
                encoding="utf-8")
            data = rank_postings(profile, tmp)
            self.assertEqual(data["ranked"][-1]["title"], "Mystery Role")
            self.assertIsNone(data["ranked"][-1]["score"])
            self.assertEqual(data["ranked"][-1]["verdict"], "unknown")

    def test_top_gaps_limited_to_three(self):
        profile = example_profile()
        data = rank_postings(profile, EXAMPLES_DIR, include_applied=True)
        ml_row = data["ranked"][2]
        self.assertEqual(ml_row["required_missing"],
                         ["TensorFlow", "PyTorch", "Machine Learning",
                          "Statistics"])
        self.assertEqual(ml_row["top_gaps"],
                         ["TensorFlow", "PyTorch", "Machine Learning"])

    def test_min_score_drops_low_and_unmeasurable(self):
        profile = example_profile()
        data = rank_postings(profile, EXAMPLES_DIR, min_score=0.5,
                             include_applied=True)
        self.assertEqual(len(data["ranked"]), 2)
        self.assertEqual(data["n_scored"], 3)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "plain.md").write_text(
                "# Mystery Role at Fictional Co\n\nNo skills listed.\n",
                encoding="utf-8")
            data = rank_postings(profile, tmp, min_score=0.5)
            self.assertEqual(data["ranked"], [])

    def test_applied_excluded_by_default_and_flagged_when_included(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "applications.jsonl"
            entry = add_application(log, company="Northwind Analytics",
                                    title="Data Analyst")
            data = rank_postings(profile, EXAMPLES_DIR, entries=[entry])
            self.assertEqual([row["title"] for row in data["ranked"]],
                             ["Business Intelligence Analyst",
                              "Machine Learning Engineer"])
            data = rank_postings(profile, EXAMPLES_DIR, entries=[entry],
                                 include_applied=True)
            northwind = data["ranked"][0]
            self.assertTrue(northwind["applied"])
            self.assertEqual(northwind["applied_id"], "APP-001")

    def test_applied_match_is_case_insensitive(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "applications.jsonl"
            entry = add_application(log, company="NORTHWIND ANALYTICS",
                                    title="DATA ANALYST")
            data = rank_postings(profile, EXAMPLES_DIR, entries=[entry],
                                 include_applied=True)
            self.assertTrue(data["ranked"][0]["applied"])

    def test_unreadable_file_is_skipped_not_fatal(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.md").write_bytes(b"\xff\xfe not utf-8 \x00")
            (Path(tmp) / "notes.json").write_text(
                '{"not": "a posting"}', encoding="utf-8")
            (Path(tmp) / "good.md").write_text(
                "# Data Analyst at Fictional Co\n\n## Requirements\n\n- SQL\n",
                encoding="utf-8")
            data = rank_postings(profile, tmp)
            # Only .md/.txt files count; the .json file is ignored entirely.
            self.assertEqual(data["n_files"], 2)
            self.assertEqual(data["n_skipped"], 1)
            self.assertEqual(data["skipped"][0]["file"], "broken.md")
            self.assertEqual(len(data["ranked"]), 1)

    def test_missing_directory_raises(self):
        profile = example_profile()
        with self.assertRaises(ValueError):
            rank_postings(profile, "/nonexistent-postings-dir")

    def test_rank_json_round_trip(self):
        profile = example_profile()
        data = rank_postings(profile, EXAMPLES_DIR, include_applied=True)
        parsed = json.loads(rank_json(data))
        self.assertEqual(parsed["ranked"][0]["company"],
                         "Northwind Analytics")
        self.assertEqual(parsed["ranked"][0]["score"], 0.625)

    def test_render_names_best_next_application(self):
        profile = example_profile()
        data = rank_postings(profile, EXAMPLES_DIR, include_applied=True)
        text = render_rank_markdown(data)
        self.assertIn("Best next application: Data Analyst — "
                      "Northwind Analytics", text)
        self.assertIn("required 5/6", text)
        self.assertIn("top gaps: Communication", text)

    def test_render_empty_and_all_applied(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            text = render_rank_markdown(rank_postings(profile, tmp))
            self.assertIn("No postings to rank", text)
            log = Path(tmp) / "applications.jsonl"
            entries = [add_application(log, company="Northwind Analytics",
                                       title="Data Analyst"),
                       add_application(log, company="Harbour Analytics",
                                       title="Business Intelligence Analyst"),
                       add_application(log, company="Vector Labs",
                                       title="Machine Learning Engineer")]
            data = rank_postings(profile, EXAMPLES_DIR, entries=entries)
            self.assertEqual(data["ranked"], [])
            text = render_rank_markdown(
                rank_postings(profile, EXAMPLES_DIR, entries=entries,
                              include_applied=True))
            self.assertIn("Best next application: none", text)
            self.assertIn("[already applied as APP-001]", text)


class TestRankCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _apply_two(self, tmp):
        for job in (EXAMPLE_JOB, EXAMPLE_JOB2):
            code, _ = self._run(["apply", "--profile", str(EXAMPLE_PROFILE),
                                 "--posting", str(job), "--out-dir", tmp])
            self.assertEqual(code, 0)
        return Path(tmp) / "applications.jsonl"

    def test_rank_cli_include_applied_and_json(self):
        code, out = self._run(["rank", "--profile", str(EXAMPLE_PROFILE),
                               "--postings-dir", str(EXAMPLES_DIR),
                               "--include-applied", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(len(data["ranked"]), 3)
        self.assertEqual(data["ranked"][0]["title"], "Data Analyst")

    def test_rank_cli_default_excludes_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._apply_two(tmp)
            code, out = self._run(["rank", "--profile",
                                   str(EXAMPLE_PROFILE), "--postings-dir",
                                   str(EXAMPLES_DIR), "--log", str(log)])
            self.assertEqual(code, 0)
            self.assertIn("Machine Learning Engineer", out)
            self.assertNotIn("Northwind Analytics", out)
            self.assertNotIn("Harbour Analytics", out)

    def test_rank_cli_out_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "rank.md"
            code, out = self._run(["rank", "--profile",
                                   str(EXAMPLE_PROFILE), "--postings-dir",
                                   str(EXAMPLES_DIR), "--include-applied",
                                   "--out", str(out_path)])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Best next application", text)
            self.assertIn("Vector Labs", text)

    def test_rank_cli_min_score_out_of_range_exit_1(self):
        code, _ = self._run(["rank", "--profile", str(EXAMPLE_PROFILE),
                             "--postings-dir", str(EXAMPLES_DIR),
                             "--min-score", "1.5"])
        self.assertEqual(code, 1)

    def test_rank_cli_missing_dir_exit_2(self):
        code, _ = self._run(["rank", "--profile", str(EXAMPLE_PROFILE),
                             "--postings-dir", "/nonexistent-postings-dir"])
        self.assertEqual(code, 2)

    def test_rank_cli_missing_profile_exit_2(self):
        code, _ = self._run(["rank", "--profile", "/nonexistent.json",
                             "--postings-dir", str(EXAMPLES_DIR)])
        self.assertEqual(code, 2)

    def test_rank_cli_corrupt_log_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "bad.jsonl"
            log.write_text("{not json\n", encoding="utf-8")
            code, _ = self._run(["rank", "--profile",
                                 str(EXAMPLE_PROFILE), "--postings-dir",
                                 str(EXAMPLES_DIR), "--log", str(log)])
            self.assertEqual(code, 2)


class TestSalaryAnalysis(unittest.TestCase):
    def test_examples_sorted_best_paying_first(self):
        profile = example_profile()
        data = analyze_salaries(profile, EXAMPLES_DIR)
        self.assertEqual(data["n_files"], 3)
        self.assertEqual(data["n_scored"], 3)
        self.assertEqual(data["n_skipped"], 0)
        titles = [row["title"] for row in data["rows"]]
        self.assertEqual(titles, ["Machine Learning Engineer",
                                  "Business Intelligence Analyst",
                                  "Data Analyst"])
        self.assertEqual([row["verdict"] for row in data["rows"]],
                         ["above target", "meets target", "meets target"])
        self.assertEqual([row["midpoint"] for row in data["rows"]],
                         [145000, 117500, 107500])

    def test_examples_medians(self):
        profile = example_profile()
        data = analyze_salaries(profile, EXAMPLES_DIR)
        self.assertEqual(set(data["medians"]), {"CAD"})
        med = data["medians"]["CAD"]
        self.assertEqual(med["n"], 3)
        self.assertEqual(med["median_min"], 105000)
        self.assertEqual(med["median_max"], 130000)
        self.assertEqual(med["median_mid"], 117500)
        self.assertEqual(data["counts"]["with_range"], 3)
        self.assertEqual(data["counts"]["meets_target"], 2)
        self.assertEqual(data["counts"]["above_target"], 1)
        self.assertEqual(data["counts"]["below_target"], 0)
        self.assertEqual(data["counts"]["not_compared"], 0)

    def test_target_from_profile(self):
        profile = example_profile()
        target = target_from_profile(profile)
        self.assertEqual(target, {"min": 95000, "max": 120000,
                                  "currency": "CAD"})
        no_target = copy.deepcopy(profile)
        no_target["preferences"] = {}
        self.assertIsNone(target_from_profile(no_target))
        min_only = copy.deepcopy(profile)
        del min_only["preferences"]["salary_max"]
        self.assertEqual(target_from_profile(min_only),
                         {"min": 95000, "max": None, "currency": "CAD"})

    def test_below_target(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "low.md").write_text(
                "# Analyst at Fictional Co\n\nSalary: $50,000 - $60,000 CAD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            self.assertEqual(data["rows"][0]["verdict"], "below target")
            self.assertEqual(data["counts"]["below_target"], 1)

    def test_min_only_target(self):
        profile = example_profile()
        del profile["preferences"]["salary_max"]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "low.md").write_text(
                "# Analyst at Fictional Co\n\nSalary: $50,000 - $60,000 CAD\n",
                encoding="utf-8")
            (Path(tmp) / "high.md").write_text(
                "# Analyst at Other Co\n\nSalary: $90,000 - $150,000 CAD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            verdicts = {r["company"]: r["verdict"] for r in data["rows"]}
            self.assertEqual(verdicts["Fictional Co"], "below target")
            self.assertEqual(verdicts["Other Co"], "meets target")

    def test_currency_mismatch_not_compared(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "usd.md").write_text(
                "# Analyst at US Co\n\nSalary: $90,000 - $110,000 USD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            row = data["rows"][0]
            self.assertEqual(row["verdict"], "not compared")
            self.assertEqual(row["verdict_reason"],
                             "posting is USD, target is CAD")
            # USD ranges still get their own median group; they are never
            # folded into the CAD medians.
            self.assertEqual(set(data["medians"]), {"USD"})
            self.assertEqual(data["medians"]["USD"]["median_mid"], 100000)

    def test_bare_dollar_not_compared(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bare.md").write_text(
                "# Analyst at Fictional Co\n\nSalary: $95,000 - $120,000\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            row = data["rows"][0]
            self.assertEqual(row["verdict"], "not compared")
            self.assertEqual(row["verdict_reason"],
                             "posting does not state a currency code")
            self.assertEqual(data["medians"], {})

    def test_no_range_listed_sorts_last(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "plain.md").write_text(
                "# Mystery Role at Fictional Co\n\nNo skills listed at all.\n",
                encoding="utf-8")
            (Path(tmp) / "paid.md").write_text(
                "# Analyst at Fictional Co\n\nSalary: $95,000 - $120,000 CAD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            last = data["rows"][-1]
            self.assertEqual(last["title"], "Mystery Role")
            self.assertEqual(last["verdict"], "no range listed")
            self.assertIsNone(last["midpoint"])
            self.assertEqual(data["counts"]["without_range"], 1)

    def test_hourly_listed_never_compared(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "hourly.md").write_text(
                "# Analyst at Fictional Co\n\nSalary: $45 - $55 per hour\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            row = data["rows"][0]
            self.assertEqual(row["period"], "hourly")
            self.assertEqual(row["verdict"], "hourly — not compared")
            self.assertEqual(data["medians"], {})
            self.assertEqual(data["counts"]["hourly"], 1)

    def test_no_target_reports_without_comparing(self):
        profile = example_profile()
        profile["preferences"] = {}
        data = analyze_salaries(profile, EXAMPLES_DIR)
        self.assertTrue(all(r["verdict"] == "no target set"
                            for r in data["rows"]))
        self.assertEqual(data["medians"]["CAD"]["median_mid"], 117500)
        text = render_salary_markdown(data)
        self.assertIn("No salary target in the profile", text)

    def test_median_even_count(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text(
                "# A Analyst at Fictional Co\n\nSalary: $100,000 - $110,000 CAD\n",
                encoding="utf-8")
            (Path(tmp) / "b.md").write_text(
                "# B Analyst at Fictional Co\n\nSalary: $120,000 - $130,000 CAD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            med = data["medians"]["CAD"]
            self.assertEqual(med["median_min"], 110000)
            self.assertEqual(med["median_max"], 120000)
            self.assertEqual(med["median_mid"], 115000)

    def test_min_pay_drops_only_judgeable_rows(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "low.md").write_text(
                "# Analyst at Low Co\n\nSalary: $50,000 - $60,000 CAD\n",
                encoding="utf-8")
            (Path(tmp) / "plain.md").write_text(
                "# Mystery Role at Fictional Co\n\nNo skills listed at all.\n",
                encoding="utf-8")
            (Path(tmp) / "usd.md").write_text(
                "# Analyst at US Co\n\nSalary: $200,000 - $250,000 USD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp, min_pay=100000)
            titles = [row["title"] for row in data["rows"]]
            # The CAD row below the floor is dropped; the no-range row
            # and the USD row cannot be judged in CAD terms, so they stay.
            self.assertEqual(titles, ["Analyst", "Mystery Role"])
            self.assertEqual(data["n_scored"], 3)

    def test_applied_flagged_with_id(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "applications.jsonl"
            entry = add_application(log, company="Vector Labs",
                                    title="Machine Learning Engineer")
            data = analyze_salaries(profile, EXAMPLES_DIR, entries=[entry])
            vector = data["rows"][0]
            self.assertTrue(vector["applied"])
            self.assertEqual(vector["applied_id"], "APP-001")
            self.assertEqual(data["counts"]["applied"], 1)

    def test_unreadable_file_is_skipped_not_fatal(self):
        profile = example_profile()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "broken.md").write_bytes(b"\xff\xfe not utf-8 \x00")
            (Path(tmp) / "good.md").write_text(
                "# Analyst at Fictional Co\n\nSalary: $95,000 - $120,000 CAD\n",
                encoding="utf-8")
            data = analyze_salaries(profile, tmp)
            self.assertEqual(data["n_files"], 2)
            self.assertEqual(data["n_skipped"], 1)
            self.assertEqual(data["skipped"][0]["file"], "broken.md")
            self.assertEqual(len(data["rows"]), 1)

    def test_missing_directory_raises(self):
        profile = example_profile()
        with self.assertRaises(ValueError):
            analyze_salaries(profile, "/nonexistent-postings-dir")

    def test_salary_json_round_trip(self):
        profile = example_profile()
        data = analyze_salaries(profile, EXAMPLES_DIR)
        parsed = json.loads(salary_json(data))
        self.assertEqual(parsed["rows"][0]["company"], "Vector Labs")
        self.assertEqual(parsed["medians"]["CAD"]["median_max"], 130000)

    def test_render_states_target_medians_and_assumption(self):
        profile = example_profile()
        data = analyze_salaries(profile, EXAMPLES_DIR)
        text = render_salary_markdown(data)
        self.assertIn("Target: CAD 95,000–120,000", text)
        self.assertIn("CAD (3 with a range): median 105,000–130,000, "
                      "median midpoint 117,500", text)
        self.assertIn("Currencies are never converted", text)
        self.assertIn("2 meet the target, 1 above the target.", text)
        self.assertIn("Machine Learning Engineer — Vector Labs  "
                      "CAD 130,000–160,000 (mid 145,000)  above target",
                      text)


class TestSalaryCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_salary_cli_examples(self):
        code, out = self._run(["salary", "--profile", str(EXAMPLE_PROFILE),
                               "--postings-dir", str(EXAMPLES_DIR)])
        self.assertEqual(code, 0)
        self.assertIn("Target: CAD 95,000–120,000", out)
        self.assertIn("median 105,000–130,000", out)
        self.assertIn("above target", out)

    def test_salary_cli_json(self):
        code, out = self._run(["salary", "--profile", str(EXAMPLE_PROFILE),
                               "--postings-dir", str(EXAMPLES_DIR),
                               "--json"])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(len(data["rows"]), 3)
        self.assertEqual(data["rows"][0]["title"],
                         "Machine Learning Engineer")

    def test_salary_cli_out_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "salary.md"
            code, out = self._run(["salary", "--profile",
                                   str(EXAMPLE_PROFILE), "--postings-dir",
                                   str(EXAMPLES_DIR),
                                   "--out", str(out_path)])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Salary analysis", text)
            self.assertIn("Vector Labs", text)

    def test_salary_cli_min_pay_negative_exit_1(self):
        code, _ = self._run(["salary", "--profile", str(EXAMPLE_PROFILE),
                             "--postings-dir", str(EXAMPLES_DIR),
                             "--min-pay", "-1"])
        self.assertEqual(code, 1)

    def test_salary_cli_min_pay_without_target_currency_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            profile = example_profile()
            del profile["preferences"]["currency"]
            profile_path.write_text(json.dumps(profile), encoding="utf-8")
            code, _ = self._run(["salary", "--profile", str(profile_path),
                                 "--postings-dir", str(EXAMPLES_DIR),
                                 "--min-pay", "100000"])
            self.assertEqual(code, 1)

    def test_salary_cli_missing_dir_exit_2(self):
        code, _ = self._run(["salary", "--profile", str(EXAMPLE_PROFILE),
                             "--postings-dir", "/nonexistent-postings-dir"])
        self.assertEqual(code, 2)

    def test_salary_cli_missing_profile_exit_2(self):
        code, _ = self._run(["salary", "--profile", "/nonexistent.json",
                             "--postings-dir", str(EXAMPLES_DIR)])
        self.assertEqual(code, 2)

    def test_salary_cli_corrupt_log_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "bad.jsonl"
            log.write_text("{not json\n", encoding="utf-8")
            code, _ = self._run(["salary", "--profile",
                                 str(EXAMPLE_PROFILE), "--postings-dir",
                                 str(EXAMPLES_DIR), "--log", str(log)])
            self.assertEqual(code, 2)


class TestOffer(unittest.TestCase):
    def _offer(self, **overrides):
        data = {"base": 130000, "currency": "CAD"}
        data.update(overrides)
        return validate_offer(data, today="2026-11-08")

    def _entries(self):
        northwind = self._offer(base=130000, currency="CAD", bonus=10000,
                                equity=20000, vesting_years=4, pto_days=20,
                                work_mode="remote",
                                deadline="2026-11-14")
        harbour = self._offer(base=105000, currency="CAD", bonus=5000,
                              work_mode="hybrid", deadline="2026-11-20")
        return [
            {"id": "APP-001", "company": "Northwind Analytics",
             "title": "Data Analyst", "status": "interview",
             "offers": [northwind]},
            {"id": "APP-002", "company": "Harbour Analytics",
             "title": "Business Intelligence Analyst", "status": "offer",
             "offers": [harbour]},
        ]

    def test_validate_offer_annual_defaults(self):
        offer = self._offer()
        self.assertEqual(offer["base"], 130000)
        self.assertEqual(offer["base_annual"], 130000)
        self.assertEqual(offer["period"], "annual")
        self.assertEqual(offer["currency"], "CAD")
        self.assertIsNone(offer["bonus"])
        self.assertIsNone(offer["equity"])
        self.assertEqual(offer["recorded_at"], "2026-11-08")

    def test_validate_offer_hourly_annualises_at_2080(self):
        offer = self._offer(base=50, period="hourly")
        self.assertEqual(offer["base_annual"], 50 * 2080)

    def test_validate_offer_monthly_annualises_at_12(self):
        offer = self._offer(base=10000, period="monthly")
        self.assertEqual(offer["base_annual"], 120000)

    def test_validate_offer_reports_every_error_at_once(self):
        with self.assertRaises(OfferError) as caught:
            validate_offer({"period": "weekly", "bonus": -5},
                           today="2026-11-08")
        message = str(caught.exception)
        self.assertIn("base is required", message)
        self.assertIn("currency is required", message)
        self.assertIn("period must be", message)
        self.assertIn("bonus must be 0 or more", message)

    def test_validate_offer_bad_values(self):
        with self.assertRaises(OfferError) as caught:
            validate_offer({"base": 0, "currency": "cad",
                            "vesting_years": 0, "pto_days": -1,
                            "work_mode": "office",
                            "start_date": "2026-13-01"},
                           today="2026-11-08")
        message = str(caught.exception)
        self.assertIn("base must be a number above 0", message)
        self.assertIn("currency must be a 3-letter code", message)
        self.assertIn("vesting_years must be", message)
        self.assertIn("pto_days must be", message)
        self.assertIn("work_mode must be", message)
        self.assertIn("start_date must be YYYY-MM-DD", message)

    def test_total_cash_and_first_year(self):
        offer = self._offer(bonus=10000, equity=20000, vesting_years=4)
        self.assertEqual(total_cash(offer), 140000)
        total, counted = first_year_total(offer)
        self.assertEqual(total, 145000)
        self.assertTrue(counted)

    def test_equity_without_vesting_is_listed_not_counted(self):
        offer = self._offer(bonus=10000, equity=20000)
        total, counted = first_year_total(offer)
        self.assertEqual(total, 140000)
        self.assertFalse(counted)

    def test_compare_ranks_by_first_year_total(self):
        profile = example_profile()
        data = compare_offers(self._entries(), profile=profile,
                              today=TODAY)
        self.assertEqual(data["counts"]["applications_with_offers"], 2)
        self.assertEqual(data["counts"]["offers"], 2)
        self.assertEqual(data["counts"]["comparable"], 2)
        rows = data["rows"]
        self.assertEqual(rows[0]["id"], "APP-001")
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["first_year_total"], 145000)
        self.assertEqual(rows[0]["total_cash"], 140000)
        self.assertEqual(rows[0]["verdict"], "above target")
        self.assertEqual(rows[1]["id"], "APP-002")
        self.assertEqual(rows[1]["rank"], 2)
        self.assertEqual(rows[1]["first_year_total"], 110000)
        self.assertEqual(rows[1]["verdict"], "meets target")
        self.assertEqual(data["best"]["id"], "APP-001")
        self.assertEqual(data["counts"]["above_target"], 1)
        self.assertEqual(data["counts"]["meets_target"], 1)

    def test_compare_below_target(self):
        profile = example_profile()
        entries = [{"id": "APP-001", "company": "Low Co",
                    "title": "Analyst", "status": "offer",
                    "offers": [self._offer(base=80000)]}]
        data = compare_offers(entries, profile=profile, today=TODAY)
        self.assertEqual(data["rows"][0]["verdict"], "below target")

    def test_compare_currency_mismatch_not_compared(self):
        profile = example_profile()
        entries = [{"id": "APP-001", "company": "US Co",
                    "title": "Analyst", "status": "offer",
                    "offers": [self._offer(base=130000, currency="USD")]}]
        data = compare_offers(entries, profile=profile, today=TODAY)
        row = data["rows"][0]
        self.assertEqual(row["verdict"], "not compared")
        self.assertIn("USD", row["verdict_reason"])
        self.assertIn("CAD", row["verdict_reason"])
        self.assertIsNone(row["rank"])
        self.assertIsNone(data["best"])
        self.assertEqual(data["counts"]["not_compared"], 1)

    def test_compare_no_target_lists_without_rank(self):
        data = compare_offers(self._entries(), profile=None, today=TODAY)
        self.assertEqual([r["verdict"] for r in data["rows"]],
                         ["no target set", "no target set"])
        self.assertEqual([r["rank"] for r in data["rows"]], [None, None])
        self.assertIsNone(data["best"])
        self.assertEqual(data["counts"]["no_target"], 2)

    def test_compare_uses_latest_offer(self):
        first = self._offer(base=120000)
        second = validate_offer({"base": 135000, "currency": "CAD"},
                                today="2026-11-10")
        entries = [{"id": "APP-001", "company": "Northwind Analytics",
                    "title": "Data Analyst", "status": "offer",
                    "offers": [first, second]}]
        data = compare_offers(entries, profile=example_profile(),
                              today=TODAY)
        row = data["rows"][0]
        self.assertEqual(row["offer"]["base"], 135000)
        self.assertEqual(row["n_offers"], 2)
        self.assertEqual(latest_offer(entries[0])["base"], 135000)

    def test_compare_deadline_flags(self):
        entries = self._entries()
        data = compare_offers(entries, profile=example_profile(),
                              today=TODAY)
        rows = {r["id"]: r for r in data["rows"]}
        self.assertEqual(rows["APP-001"]["days_to_deadline"], 6)
        self.assertFalse(rows["APP-001"]["deadline_passed"])
        self.assertEqual(rows["APP-002"]["days_to_deadline"], 12)
        past = [{"id": "APP-003", "company": "Late Co", "title": "Analyst",
                 "status": "offer",
                 "offers": [self._offer(deadline="2026-11-01")]}]
        data = compare_offers(past, profile=example_profile(), today=TODAY)
        self.assertTrue(data["rows"][0]["deadline_passed"])
        self.assertEqual(data["rows"][0]["days_to_deadline"], -7)
        self.assertEqual(data["counts"]["deadline_passed"], 1)

    def test_compare_no_offers_empty(self):
        data = compare_offers([{"id": "APP-001", "company": "No Offer Co",
                                "title": "Analyst", "status": "applied",
                                "offers": []}],
                              profile=example_profile(), today=TODAY)
        self.assertEqual(data["rows"], [])
        self.assertEqual(data["counts"]["applications_with_offers"], 0)
        self.assertIsNone(data["best"])
        self.assertIn("No offers recorded yet",
                      render_offer_markdown(data))

    def test_render_marks_equity_without_vesting(self):
        entries = [{"id": "APP-001", "company": "Northwind Analytics",
                    "title": "Data Analyst", "status": "offer",
                    "offers": [self._offer(equity=20000)]}]
        data = compare_offers(entries, profile=example_profile(),
                              today=TODAY)
        text = render_offer_markdown(data)
        self.assertIn("equity CAD 20,000", text)
        self.assertIn("not counted: no vesting schedule", text)
        self.assertIn("130,000 first year", text)  # equity excluded
        self.assertIn("Best comparable offer", text)

    def test_render_hourly_base_shows_stated_figure(self):
        entries = [{"id": "APP-001", "company": "Hourly Co",
                    "title": "Analyst", "status": "offer",
                    "offers": [self._offer(base=50, period="hourly")]}]
        data = compare_offers(entries, profile=example_profile(),
                              today=TODAY)
        text = render_offer_markdown(data)
        self.assertIn("as stated CAD 50/hourly", text)
        self.assertIn("104,000", text)

    def test_record_offer_appends_and_allows_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            entry = record_offer(log, "APP-001", self._offer(),
                                 today="2026-11-08")
            self.assertEqual(len(entry["offers"]), 1)
            self.assertEqual(entry["history"][-1],
                             {"kind": "offer", "at": "2026-11-08"})
            entry = record_offer(
                log, "APP-001",
                validate_offer({"base": 135000, "currency": "CAD"},
                               today="2026-11-10"),
                today="2026-11-10")
            self.assertEqual(len(entry["offers"]), 2)
            self.assertEqual(read_log(log)[0]["status"], "applied")

    def test_record_offer_refuses_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            set_status(log, "APP-001", "rejected", today="2026-11-05")
            with self.assertRaises(TrackerError):
                record_offer(log, "APP-001", self._offer(),
                             today="2026-11-08")

    def test_record_offer_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            with self.assertRaises(TrackerError):
                record_offer(log, "APP-999", self._offer(),
                             today="2026-11-08")


class TestOfferCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _log_with_offers(self, tmp):
        log = Path(tmp) / "log.jsonl"
        add_application(log, "Northwind Analytics", "Data Analyst",
                        today="2026-11-01")
        add_application(log, "Harbour Analytics",
                        "Business Intelligence Analyst", today="2026-11-01")
        return log

    def test_offer_cli_record_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, out = self._run(["offer", "--record", "--id", "APP-001",
                                   "--log", str(log),
                                   "--base", "130000", "--currency", "CAD",
                                   "--bonus", "10000", "--equity", "20000",
                                   "--vesting-years", "4", "--pto-days", "20",
                                   "--mode", "remote",
                                   "--deadline", "2026-11-14",
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            self.assertIn("recorded an offer for APP-001", out)
            code, out = self._run(["offer", "--record", "--id", "APP-002",
                                   "--log", str(log),
                                   "--base", "105000", "--currency", "CAD",
                                   "--bonus", "5000", "--mode", "hybrid",
                                   "--deadline", "2026-11-20",
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            code, out = self._run(["offer", "--profile",
                                   str(EXAMPLE_PROFILE), "--log", str(log),
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            self.assertIn("Best comparable offer", out)
            self.assertIn("Northwind Analytics", out)
            self.assertIn("145,000 first year", out)
            self.assertIn("above target", out)
            self.assertIn("meets target", out)
            self.assertIn("decides in 6d", out)

    def test_offer_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            self._run(["offer", "--record", "--id", "APP-001",
                       "--log", str(log), "--base", "130000",
                       "--currency", "CAD", "--today", "2026-11-08"])
            code, out = self._run(["offer", "--profile",
                                   str(EXAMPLE_PROFILE), "--log", str(log),
                                   "--json", "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            data = json.loads(out)
            self.assertEqual(len(data["rows"]), 1)
            self.assertEqual(data["rows"][0]["rank"], 1)
            self.assertEqual(data["best"]["id"], "APP-001")

    def test_offer_cli_out_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            out_path = Path(tmp) / "offers.md"
            code, out = self._run(["offer", "--profile",
                                   str(EXAMPLE_PROFILE), "--log", str(log),
                                   "--out", str(out_path)])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            self.assertIn("Offer comparison",
                          out_path.read_text(encoding="utf-8"))

    def test_offer_cli_no_offers(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, out = self._run(["offer", "--profile",
                                   str(EXAMPLE_PROFILE), "--log", str(log)])
            self.assertEqual(code, 0)
            self.assertIn("No offers recorded yet", out)

    def test_offer_cli_record_bad_base_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, _ = self._run(["offer", "--record", "--id", "APP-001",
                                 "--log", str(log), "--base", "-5",
                                 "--currency", "CAD"])
            self.assertEqual(code, 1)
            self.assertEqual(read_log(log)[0].get("offers") or [], [])

    def test_offer_cli_record_missing_currency_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, _ = self._run(["offer", "--record", "--id", "APP-001",
                                 "--log", str(log), "--base", "130000"])
            self.assertEqual(code, 1)

    def test_offer_cli_record_without_id_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, _ = self._run(["offer", "--record", "--log", str(log),
                                 "--base", "130000", "--currency", "CAD"])
            self.assertEqual(code, 1)

    def test_offer_cli_record_flags_without_record_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, _ = self._run(["offer", "--log", str(log),
                                 "--base", "130000", "--currency", "CAD"])
            self.assertEqual(code, 1)

    def test_offer_cli_record_terminal_exit_6(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            set_status(log, "APP-001", "rejected", today="2026-11-05")
            code, _ = self._run(["offer", "--record", "--id", "APP-001",
                                 "--log", str(log), "--base", "130000",
                                 "--currency", "CAD"])
            self.assertEqual(code, 6)

    def test_offer_cli_record_unknown_id_exit_6(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, _ = self._run(["offer", "--record", "--id", "APP-999",
                                 "--log", str(log), "--base", "130000",
                                 "--currency", "CAD"])
            self.assertEqual(code, 6)

    def test_offer_cli_bad_today_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_offers(tmp)
            code, _ = self._run(["offer", "--profile",
                                 str(EXAMPLE_PROFILE), "--log", str(log),
                                 "--today", "2026-13-40"])
            self.assertEqual(code, 1)


class TestInterview(unittest.TestCase):
    def _round(self, **overrides):
        data = {"date": "2026-11-10", "round": "Hiring manager",
                "interviewer": "Priya Nair", "mode": "video",
                "duration_minutes": 45, "notes": "Bring the dashboard story."}
        data.update(overrides)
        return data

    def _entries(self):
        return [
            {"id": "APP-001", "company": "Northwind Analytics",
             "title": "Data Analyst", "status": "interview",
             "interviews": [
                 self._round(date="2026-11-10", round="Hiring manager"),
                 self._round(date="2026-11-06", round="Recruiter screen",
                             interviewer=None, mode="phone",
                             duration_minutes=30, notes=None),
                 self._round(date="2026-11-08", round="Panel",
                             interviewer="Sam Ortiz", mode="onsite",
                             duration_minutes=60, notes=None),
             ]},
            {"id": "APP-002", "company": "Harbour Analytics",
             "title": "Business Intelligence Analyst", "status": "screening",
             "interviews": [
                 self._round(date="2026-11-09", round="Recruiter screen",
                             interviewer=None, mode="phone",
                             duration_minutes=30, notes=None),
             ]},
            {"id": "APP-003", "company": "Vector Labs",
             "title": "Machine Learning Engineer", "status": "applied"},
        ]

    def test_validate_interview_valid_defaults(self):
        interview = validate_interview(
            {"date": "2026-11-10", "round": "  Hiring manager  ",
             "interviewer": "  Priya Nair "},
            today="2026-11-08")
        self.assertEqual(interview["date"], "2026-11-10")
        self.assertEqual(interview["round"], "Hiring manager")
        self.assertEqual(interview["interviewer"], "Priya Nair")
        self.assertIsNone(interview["mode"])
        self.assertIsNone(interview["duration_minutes"])
        self.assertIsNone(interview["notes"])
        self.assertEqual(interview["recorded_at"], "2026-11-08")

    def test_validate_interview_minimal(self):
        interview = validate_interview(
            {"date": "2026-11-10", "round": "Panel"}, today="2026-11-08")
        self.assertEqual(interview["round"], "Panel")
        self.assertIsNone(interview["interviewer"])

    def test_validate_interview_reports_every_error_at_once(self):
        with self.assertRaises(InterviewError) as caught:
            validate_interview({"mode": "carrier pigeon",
                                "duration_minutes": 0, "notes": 42,
                                "interviewer": 7})
        message = str(caught.exception)
        self.assertIn("date is required", message)
        self.assertIn("round is required", message)
        self.assertIn("interviewer must be text", message)
        self.assertIn("mode must be one of phone, video, onsite", message)
        self.assertIn("duration_minutes must be a whole number above 0",
                      message)
        self.assertIn("notes must be text", message)

    def test_validate_interview_bad_date(self):
        with self.assertRaises(InterviewError):
            validate_interview({"date": "2026-13-01", "round": "Panel"})

    def test_validate_interview_bad_mode(self):
        with self.assertRaises(InterviewError):
            validate_interview(self._round(mode="telepathy"))

    def test_validate_interview_bad_duration(self):
        for bad in (0, -30, "45", 45.5, True):
            with self.subTest(bad=bad):
                with self.assertRaises(InterviewError):
                    validate_interview(self._round(duration_minutes=bad))

    def test_validate_interview_blank_round(self):
        with self.assertRaises(InterviewError):
            validate_interview({"date": "2026-11-10", "round": "   "})

    def test_collect_splits_upcoming_and_past(self):
        data = collect_interviews(self._entries(), today=TODAY)
        self.assertEqual(data["counts"], {"upcoming": 3, "past": 1,
                                          "today": 1})
        self.assertEqual([r["round"] for r in data["upcoming"]],
                         ["Panel", "Recruiter screen", "Hiring manager"])
        self.assertEqual(data["upcoming"][0]["days_until"], 0)
        self.assertEqual(data["upcoming"][1]["days_until"], 1)
        self.assertEqual(data["upcoming"][2]["days_until"], 2)
        self.assertEqual(data["past"][0]["round"], "Recruiter screen")
        self.assertEqual(data["past"][0]["days_ago"], 2)

    def test_collect_upcoming_sorted_soonest_first(self):
        entries = self._entries()
        entries[0]["interviews"] = list(reversed(entries[0]["interviews"]))
        data = collect_interviews(entries, today=TODAY)
        self.assertEqual([r["date"] for r in data["upcoming"]],
                         ["2026-11-08", "2026-11-09", "2026-11-10"])

    def test_collect_past_sorted_most_recent_first(self):
        entries = self._entries()
        entries[0]["interviews"].append(
            self._round(date="2026-11-04", round="Intro call",
                        interviewer=None, mode="phone",
                        duration_minutes=15, notes=None))
        data = collect_interviews(entries, today=TODAY)
        self.assertEqual([r["date"] for r in data["past"]],
                         ["2026-11-06", "2026-11-04"])

    def test_collect_skips_unparseable_date(self):
        entries = self._entries()
        entries[0]["interviews"].append({"date": "soon", "round": "Panel"})
        data = collect_interviews(entries, today=TODAY)
        self.assertEqual(data["counts"]["upcoming"], 3)
        self.assertEqual(data["counts"]["past"], 1)

    def test_collect_empty_log(self):
        data = collect_interviews([], today=TODAY)
        self.assertEqual(data["counts"], {"upcoming": 0, "past": 0,
                                          "today": 0})

    def test_interviews_for_unknown_id(self):
        entry, rounds = interviews_for(self._entries(), "APP-999",
                                       today=TODAY)
        self.assertIsNone(entry)
        self.assertEqual(rounds, [])

    def test_interviews_for_filters_one_application(self):
        entry, rounds = interviews_for(self._entries(), "APP-001",
                                       today=TODAY)
        self.assertEqual(entry["company"], "Northwind Analytics")
        self.assertEqual([r["round"] for r in rounds],
                         ["Recruiter screen", "Panel", "Hiring manager"])
        self.assertEqual(rounds[0]["days_ago"], 2)
        self.assertEqual(rounds[1]["days_until"], 0)
        self.assertEqual(rounds[2]["days_until"], 2)

    def test_render_marks_today_and_days(self):
        data = collect_interviews(self._entries(), today=TODAY)
        text = render_interview_markdown(data)
        self.assertIn("3 upcoming (1 today), 1 past", text)
        self.assertIn("Panel", text)
        self.assertIn("— today", text)
        self.assertIn("— in 1d", text)
        self.assertIn("— in 2d", text)
        self.assertIn("— 2d ago", text)
        self.assertIn("with Priya Nair", text)
        self.assertIn("(video)", text)
        self.assertIn("45 min", text)
        self.assertIn("notes: Bring the dashboard story.", text)

    def test_render_empty_state(self):
        data = collect_interviews([], today=TODAY)
        text = render_interview_markdown(data)
        self.assertIn("0 upcoming, 0 past", text)
        self.assertIn("No interviews recorded yet", text)

    def test_interview_json_round_trip(self):
        data = collect_interviews(self._entries(), today=TODAY)
        parsed = json.loads(interview_json(data))
        self.assertEqual(parsed["counts"]["upcoming"], 3)
        self.assertEqual(parsed["upcoming"][0]["round"], "Panel")

    def test_record_interview_appends_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            interview = validate_interview(self._round(), today="2026-11-08")
            entry = record_interview(log, "APP-001", interview,
                                     today="2026-11-08")
            self.assertEqual(len(entry["interviews"]), 1)
            self.assertEqual(entry["interviews"][0]["round"],
                             "Hiring manager")
            self.assertEqual(entry["history"][-1],
                             {"kind": "interview", "at": "2026-11-10",
                              "round": "Hiring manager"})

    def test_record_interview_allows_two_same_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            record_interview(log, "APP-001",
                             validate_interview(self._round(round="Panel"),
                                                today="2026-11-08"),
                             today="2026-11-08")
            entry = record_interview(
                log, "APP-001",
                validate_interview(self._round(round="Hiring manager"),
                                   today="2026-11-08"),
                today="2026-11-08")
            self.assertEqual(len(entry["interviews"]), 2)
            self.assertEqual([r["round"] for r in entry["interviews"]],
                             ["Panel", "Hiring manager"])

    def test_record_interview_does_not_move_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            entry = record_interview(
                log, "APP-001",
                validate_interview(self._round(), today="2026-11-08"),
                today="2026-11-08")
            self.assertEqual(entry["status"], "applied")

    def test_record_interview_refuses_terminal(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            set_status(log, "APP-001", "withdrawn", today="2026-11-05")
            with self.assertRaises(TrackerError):
                record_interview(log, "APP-001",
                                 validate_interview(self._round(),
                                                    today="2026-11-08"),
                                 today="2026-11-08")

    def test_record_interview_unknown_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "log.jsonl"
            add_application(log, "Northwind Analytics", "Data Analyst",
                            today="2026-11-01")
            with self.assertRaises(TrackerError):
                record_interview(log, "APP-999",
                                 validate_interview(self._round(),
                                                    today="2026-11-08"),
                                 today="2026-11-08")


class TestInterviewCLI(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def _log_with_rounds(self, tmp):
        log = Path(tmp) / "log.jsonl"
        add_application(log, "Northwind Analytics", "Data Analyst",
                        today="2026-11-01")
        add_application(log, "Harbour Analytics",
                        "Business Intelligence Analyst", today="2026-11-01")
        return log

    def test_interview_cli_record_and_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, out = self._run(["interview", "--record", "--id", "APP-001",
                                   "--log", str(log),
                                   "--date", "2026-11-10",
                                   "--round", "Hiring manager",
                                   "--interviewer", "Priya Nair",
                                   "--mode", "video",
                                   "--duration-minutes", "45",
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            self.assertIn("recorded an interview for APP-001", out)
            self._run(["interview", "--record", "--id", "APP-001",
                       "--log", str(log), "--date", "2026-11-06",
                       "--round", "Recruiter screen", "--mode", "phone",
                       "--today", "2026-11-08"])
            code, out = self._run(["interview", "--log", str(log),
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            self.assertIn("1 upcoming, 1 past", out)
            self.assertIn("Hiring manager", out)
            self.assertIn("— in 2d", out)
            self.assertIn("Recruiter screen", out)
            self.assertIn("— 2d ago", out)

    def test_interview_cli_list_id_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            self._run(["interview", "--record", "--id", "APP-001",
                       "--log", str(log), "--date", "2026-11-10",
                       "--round", "Hiring manager",
                       "--today", "2026-11-08"])
            self._run(["interview", "--record", "--id", "APP-002",
                       "--log", str(log), "--date", "2026-11-09",
                       "--round", "Recruiter screen",
                       "--today", "2026-11-08"])
            code, out = self._run(["interview", "--log", str(log),
                                   "--id", "APP-002",
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            self.assertIn("1 upcoming, 0 past", out)
            self.assertIn("Harbour Analytics", out)
            self.assertNotIn("Northwind Analytics", out)

    def test_interview_cli_list_id_unknown_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--log", str(log),
                                 "--id", "APP-999"])
            self.assertEqual(code, 1)

    def test_interview_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            self._run(["interview", "--record", "--id", "APP-001",
                       "--log", str(log), "--date", "2026-11-10",
                       "--round", "Hiring manager",
                       "--today", "2026-11-08"])
            code, out = self._run(["interview", "--log", str(log),
                                   "--json", "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            data = json.loads(out)
            self.assertEqual(data["counts"]["upcoming"], 1)
            self.assertEqual(data["upcoming"][0]["days_until"], 2)

    def test_interview_cli_out_writes_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            out_path = Path(tmp) / "interviews.md"
            code, out = self._run(["interview", "--log", str(log),
                                   "--out", str(out_path),
                                   "--today", "2026-11-08"])
            self.assertEqual(code, 0)
            self.assertIn("wrote", out)
            self.assertIn("Interviews —", out_path.read_text(encoding="utf-8"))

    def test_interview_cli_no_interviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, out = self._run(["interview", "--log", str(log)])
            self.assertEqual(code, 0)
            self.assertIn("No interviews recorded yet", out)

    def test_interview_cli_record_missing_round_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--record", "--id", "APP-001",
                                 "--log", str(log), "--date", "2026-11-10"])
            self.assertEqual(code, 1)
            self.assertEqual(read_log(log)[0].get("interviews") or [], [])

    def test_interview_cli_record_missing_date_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--record", "--id", "APP-001",
                                 "--log", str(log),
                                 "--round", "Hiring manager"])
            self.assertEqual(code, 1)

    def test_interview_cli_record_flags_without_record_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--log", str(log),
                                 "--date", "2026-11-10",
                                 "--round", "Hiring manager"])
            self.assertEqual(code, 1)

    def test_interview_cli_record_without_id_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--record", "--log", str(log),
                                 "--date", "2026-11-10",
                                 "--round", "Hiring manager"])
            self.assertEqual(code, 1)

    def test_interview_cli_record_terminal_exit_6(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            set_status(log, "APP-001", "rejected", today="2026-11-05")
            code, _ = self._run(["interview", "--record", "--id", "APP-001",
                                 "--log", str(log), "--date", "2026-11-10",
                                 "--round", "Hiring manager"])
            self.assertEqual(code, 6)

    def test_interview_cli_record_unknown_id_exit_6(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--record", "--id", "APP-999",
                                 "--log", str(log), "--date", "2026-11-10",
                                 "--round", "Hiring manager"])
            self.assertEqual(code, 6)

    def test_interview_cli_bad_today_exit_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = self._log_with_rounds(tmp)
            code, _ = self._run(["interview", "--log", str(log),
                                 "--today", "2026-13-40"])
            self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
