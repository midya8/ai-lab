import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT.parent / "ragkit" / "examples" / "docs"

from docqa import DocQA  # noqa: E402
from docqa.evaluate import run_eval  # noqa: E402
from docqa.hybrid import embed_ngrams  # noqa: E402


def test_embedding_is_unit_and_typo_tolerant():
    a, b, c = (embed_ngrams(t) for t in ("restocking fee", "restocing fee", "weather forecast"))
    assert abs(np.linalg.norm(a) - 1) < 1e-5
    assert a @ b > a @ c + 0.3


def test_embedding_empty_text_is_finite():
    assert np.isfinite(embed_ngrams("")).all()


def test_hybrid_search_returns_chunks():
    qa = DocQA.from_dir(DOCS, retrieval="hybrid")
    hits = qa.retriever.search("how long to return a bike", k=3)
    assert 0 < len(hits) <= 3
    assert any("returns.md" in h.chunk.source for h in hits)


def test_hybrid_matches_bm25_on_eval_sets():
    for f in ("questions", "paraphrase", "broad"):
        base = run_eval(DocQA.from_dir(DOCS), ROOT / "eval" / f"{f}.json")
        hyb = run_eval(DocQA.from_dir(DOCS, retrieval="hybrid"), ROOT / "eval" / f"{f}.json")
        assert hyb["answered_correctly"] >= base["answered_correctly"]
        assert hyb["correctly_abstained"] >= base["correctly_abstained"]


def test_bad_retrieval_mode():
    with pytest.raises(ValueError):
        DocQA.from_dir(DOCS, retrieval="dense")
