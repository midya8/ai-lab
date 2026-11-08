import numpy as np
import pytest

from codesearch.index import Index
from codesearch.search import MODES, Searcher, query_symbols
from codesearch.store import sync


@pytest.fixture
def searcher(repo):
    chunks, _ = sync(repo, None)
    return Searcher(Index.build(chunks))


def top(s, q, **kw):
    return [h.chunk.qualname for h in s.search(q, **kw)]


def test_natural_language_finds_function_by_docstring(searcher):
    assert top(searcher, "retry a call with exponential backoff")[0] == "retry_with_backoff"
    assert top(searcher, "turn a title into a url slug")[0] == "slugify"


def test_camelcase_symbol_matches_prose_query(searcher):
    assert top(searcher, "parse http headers")[0] == "parseHTTPHeaders"


def test_inflections_match(searcher):
    assert top(searcher, "retrying failures")[0] in ("retry_with_backoff", "CircuitBreaker.record_failure")


def test_body_and_comment_terms_are_searchable(searcher):
    assert top(searcher, "os replace rename temp file")[0] == "save_json_atomically"
    assert "retry_with_backoff" in top(searcher, "wait before trying again", k=3)


def test_exact_symbol_boost_beats_text_relevance(searcher):
    hits = searcher.search("`word_count`", k=3)
    assert hits[0].chunk.qualname == "word_count" and hits[0].exact_symbol
    hits = searcher.search("how does record_failure behave", k=3)
    assert hits[0].chunk.qualname == "CircuitBreaker.record_failure" and hits[0].exact_symbol


def test_every_mode_runs_and_returns_ranked_hits(searcher):
    for m in MODES:
        hits = searcher.search("save json file", k=5, mode=m)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
        assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_unknown_mode_rejected(searcher):
    with pytest.raises(ValueError):
        searcher.search("x", mode="magic")


def test_empty_and_stopword_only_queries_return_nothing(searcher):
    assert searcher.search("") == []
    assert searcher.search("the of and") == []
    assert searcher.search("zzzzqqqq") == []


def test_filters_kind_and_path(searcher):
    assert {h.chunk.kind for h in searcher.search("breaker", kinds=["class"])} == {"class"}
    assert all(h.chunk.path.startswith("app/text") for h in searcher.search("text", path_prefix="app/text"))


def test_tests_are_demoted_unless_asked_for(searcher):
    q = "retry with backoff"
    assert not searcher.search(q, k=1)[0].chunk.is_test
    assert searcher.search("test " + q, k=1)[0].chunk.is_test  # asking for tests lifts the penalty
    demoted = next(h.score for h in searcher.search(q, k=10) if h.chunk.is_test)
    plain = next(h.score for h in searcher.search(q, k=10, include_tests=True) if h.chunk.is_test)
    assert plain == pytest.approx(demoted / 0.6)


def test_matched_terms_explain_hits(searcher):
    h = searcher.search("exponential backoff", k=1)[0]
    assert h.chunk.qualname == "retry_with_backoff"
    assert "backoff" in h.matched["name"] and "exponenti" in " ".join(h.matched["doc"]) or h.matched["doc"]


def test_query_symbols():
    assert query_symbols("use `slugify` or parse_http_headers() with fooBar and plain words") == [
        "slugify", "parse_http_headers", "fooBar"]
    assert query_symbols("just words here") == []


def test_similar_excludes_itself_and_finds_siblings(searcher):
    hits = searcher.similar("app/textutil.py::slugify", k=5)
    assert all(h.chunk.qualname != "slugify" for h in hits)
    sim = [h.chunk.qualname for h in searcher.similar("CircuitBreaker.allow", k=3)]
    assert "CircuitBreaker.record_failure" in sim or "CircuitBreaker" in sim


def test_resolve_ambiguity_and_unknown(repo):
    (repo / "app" / "other.py").write_text("def slugify(x):\n    return x\n")
    chunks, _ = sync(repo, None)
    s = Searcher(Index.build(chunks))
    with pytest.raises(KeyError, match="ambiguous"):
        s.resolve("slugify")
    assert s.resolve("app/other.py::slugify") != s.resolve("app/textutil.py::slugify")
    with pytest.raises(KeyError):
        s.resolve("nope")


def test_duplicates_found_and_tests_excluded_by_default(repo):
    body = "def {n}(rows):\n    out = []\n    for row in rows:\n        if row.get('ok'):\n            out.append(row['value'] * 2)\n    return out\n"
    (repo / "a.py").write_text(body.format(n="doubled_values"))
    (repo / "b.py").write_text(body.format(n="doubled_values_copy"))
    (repo / "tests").mkdir(exist_ok=True)
    (repo / "tests" / "test_c.py").write_text(body.format(n="doubled_in_test"))
    chunks, _ = sync(repo, None)
    s = Searcher(Index.build(chunks))
    pairs = s.duplicates(threshold=0.8, min_lines=5, min_jaccard=0.5)
    names = {frozenset((a.qualname, b.qualname)) for a, b, *_ in pairs}
    assert frozenset(("doubled_values", "doubled_values_copy")) in names
    assert not any("doubled_in_test" in n for n in names)
    with_tests = s.duplicates(threshold=0.6, min_lines=5, min_jaccard=0.5, include_tests=True)
    assert any("doubled_in_test" in (a.qualname, b.qualname) for a, b, *_ in with_tests)


def test_lsa_is_deterministic_and_unit_norm(repo):
    chunks, _ = sync(repo, None)
    a, b = Index.build(chunks).lsa_matrix(), Index.build(chunks).lsa_matrix()
    assert np.allclose(a, b)
    norms = np.linalg.norm(a, axis=1)
    assert np.all((np.abs(norms - 1) < 1e-4) | (norms == 0))


def test_tiny_corpus_does_not_crash():
    from codesearch.chunker import chunk_source
    ix = Index.build(chunk_source("def only():\n    '''the only function'''\n", "x.py"))
    s = Searcher(ix)
    assert s.search("only function", mode="lsa") == []
    assert s.search("only function")[0].chunk.qualname == "only"
    assert s.duplicates() == []
