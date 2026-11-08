import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

import pytest  # noqa: E402

from docqa import DocQA  # noqa: E402
from docqa.__main__ import main  # noqa: E402
from docqa.evaluate import run_eval  # noqa: E402
from docqa.spell import SpellCorrector, osa_distance  # noqa: E402

CORPUS = ["Returns are accepted within 30 days. Refunds go to the original payment.",
          "The shop is closed on Sundays. Shipping is free over $150. Shipping takes days."]


@pytest.fixture(scope="module")
def sc():
    return SpellCorrector(CORPUS)


@pytest.mark.parametrize("a,b,d", [
    ("return", "return", 0), ("retrun", "return", 1),  # transposition counts once
    ("retun", "return", 1), ("retturn", "return", 1), ("rexurn", "return", 1),
])
def test_osa_distance(a, b, d):
    assert osa_distance(a, b, 2) == d


def test_osa_limit_cutoff():
    assert osa_distance("abcdef", "uvwxyz", 1) == 2  # reports limit + 1
    assert osa_distance("a", "abcdef", 2) == 3


def test_known_words_and_inflections_untouched(sc):
    assert sc.suggest("Refunds") is None
    assert sc.suggest("shipped") is None  # stems to a corpus stem


def test_fixes_typo_against_corpus_only(sc):
    c = sc.suggest("refudns")
    assert c and c.corrected == "refunds" and c.distance == 1


def test_inflection_only_corpus_still_corrects_to_stem(sc):
    c = sc.suggest("retrun")  # corpus has "returns" only
    assert c and c.corrected == "returns"


def test_never_invents_vocabulary(sc):
    for w in ["capital", "france", "president", "helmets", "electric"]:
        assert sc.suggest(w) is None


def test_short_words_and_stopwords_skipped(sc):
    assert sc.suggest("sho") is None
    assert sc.suggest("which") is None
    assert sc.suggest("dcys") is None  # 4 letters: a substitution is not allowed
    assert sc.suggest("dasy").corrected == "days"  # but an adjacent swap is


def test_four_letter_swap_allowed():
    assert SpellCorrector(["The shop will open at nine."]).suggest("opne").corrected == "open"


def test_correct_preserves_case_and_reports(sc):
    fixed, fixes = sc.correct("Refudns arrive in 30 days?")
    assert fixed.startswith("Refunds arrive") and fixed.endswith("days?")
    assert [(f.original, f.corrected) for f in fixes] == [("refudns", "refunds")]


def test_tie_break_deterministic():
    s = SpellCorrector(["alpha bravo brave brave brave"])
    # 'bravi' is distance 1 from both 'bravo' and 'brave'; the commoner one wins
    assert s.suggest("bravi").corrected == "brave"


def test_numbers_and_symbols_untouched(sc):
    fixed, fixes = sc.correct("Is $150 free over 1500?")
    assert fixes == [] and fixed == "Is $150 free over 1500?"


@pytest.fixture(scope="module")
def qa():
    return DocQA.from_dir(DOCS)


def test_typo_question_answers_and_reports(qa):
    r = qa.ask("How long do I have to retrun a bike?")
    assert r.answered and "30 days" in r.text
    assert r.question == "How long do I have to retrun a bike?"  # caller's text preserved
    assert any(c.original == "retrun" for c in r.corrections)
    assert r.to_dict()["corrections"][0]["original"] == "retrun"


def test_clean_question_has_no_corrections_key(qa):
    r = qa.ask("How long do I have to return a bike?")
    assert r.corrections == [] and "corrections" not in r.to_dict()


def test_spell_off_misses_the_typo():
    # without correction the typo drops a key term; the answer (if any) is not the right one
    r = DocQA.from_dir(DOCS, spell=False).ask("How long do I have to retrun a bike?")
    assert "30 days" not in r.text and r.corrections == []


def test_typo_benchmark_lift():
    path = ROOT / "eval" / "typos.json"
    on = run_eval(DocQA.from_dir(DOCS), path)
    off = run_eval(DocQA.from_dir(DOCS, spell=False), path)
    assert off["answered_correctly"] <= 8 and on["answered_correctly"] == on["answerable"]
    assert on["abstain_rate"] == 1.0 and off["abstain_rate"] == 1.0


@pytest.mark.parametrize("name", ["questions", "paraphrase", "broad"])
def test_spell_does_not_change_clean_results(name):
    path = ROOT / "eval" / f"{name}.json"
    on = run_eval(DocQA.from_dir(DOCS), path)
    off = run_eval(DocQA.from_dir(DOCS, spell=False), path)
    assert on == off


def test_cli_notes_interpretation_and_no_spell_flag(capsys):
    assert main(["ask", str(DOCS), "How long do I have to retrun a bike?"]) == 0
    out = capsys.readouterr().out
    assert "retrun -> returns" in out and "30 days" in out
    assert main(["ask", str(DOCS), "How long do I have to retrun a bike?", "--no-spell"]) in (0, 1)
    assert "30 days" not in capsys.readouterr().out
