"""Tests for TF-IDF / BM25 retrieval."""

import unittest

from ragkit.chunk import Chunk
from ragkit.retrieve import (
    Retriever,
    ScoredChunk,
    select_for_answer,
    top_relative,
)


def make_chunks():
    return [
        Chunk("physics.md",
              "Quantum entanglement links particles across distance. "
              "Entangled photons share a single quantum state.",
              1, 2),
        Chunk("cooking.md",
              "Fold the egg whites into the batter gently. "
              "Bake the cake at 180 degrees for thirty minutes.",
              1, 2),
        Chunk("bikes.md",
              "A red bicycle parked outside the shop. "
              "The bicycle chain needs oil after rain.",
              1, 2),
        Chunk("bikes2.md",
              "The bicycle was painted red yesterday. "
              "A tune-up keeps the bicycle running well.",
              1, 2),
    ]


class TestRetriever(unittest.TestCase):
    def test_known_query_returns_known_doc_first_bm25(self):
        retriever = Retriever(make_chunks(), scoring="bm25")
        results = retriever.search("quantum entanglement photons", k=3)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "physics.md")

    def test_known_query_returns_known_doc_first_tfidf(self):
        retriever = Retriever(make_chunks(), scoring="tfidf")
        results = retriever.search("quantum entanglement photons", k=3)
        self.assertTrue(results)
        self.assertEqual(results[0].chunk.source, "physics.md")

    def test_cooking_query_returns_cooking_doc_first(self):
        retriever = Retriever(make_chunks())
        results = retriever.search("bake cake egg whites", k=3)
        self.assertEqual(results[0].chunk.source, "cooking.md")

    def test_exact_phrase_boost_ranks_phrase_chunk_first(self):
        retriever = Retriever(make_chunks())
        results = retriever.search("red bicycle", k=2)
        self.assertEqual(results[0].chunk.source, "bikes.md")

    def test_top_k_is_respected(self):
        retriever = Retriever(make_chunks())
        results = retriever.search("bicycle", k=1)
        self.assertEqual(len(results), 1)
        results = retriever.search("bicycle", k=2)
        self.assertEqual(len(results), 2)

    def test_scores_are_descending(self):
        retriever = Retriever(make_chunks())
        results = retriever.search("bicycle quantum cake", k=5)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s > 0 for s in scores))

    def test_empty_corpus_returns_no_results(self):
        retriever = Retriever([])
        self.assertEqual(retriever.search("anything", k=5), [])

    def test_unknown_terms_return_no_results(self):
        retriever = Retriever(make_chunks())
        self.assertEqual(retriever.search("zxqv wjorp", k=5), [])

    def test_empty_query_returns_no_results(self):
        retriever = Retriever(make_chunks())
        self.assertEqual(retriever.search("", k=5), [])
        self.assertEqual(retriever.search("   ", k=5), [])

    def test_search_is_deterministic(self):
        retriever = Retriever(make_chunks())
        first = [(r.chunk.source, r.score) for r in retriever.search("bicycle", k=5)]
        second = [(r.chunk.source, r.score) for r in retriever.search("bicycle", k=5)]
        self.assertEqual(first, second)

    def test_invalid_scoring_raises(self):
        with self.assertRaises(ValueError):
            Retriever(make_chunks(), scoring="word2vec")

    def test_invalid_k_raises(self):
        retriever = Retriever(make_chunks())
        with self.assertRaises(ValueError):
            retriever.search("bicycle", k=0)

    def test_top_relative_keeps_only_close_scores(self):
        chunks = make_chunks()
        scored = [
            ScoredChunk(chunks[0], 10.0),
            ScoredChunk(chunks[1], 6.0),
            ScoredChunk(chunks[2], 4.0),
        ]
        kept = top_relative(scored, ratio=0.5)
        self.assertEqual([s.chunk.source for s in kept],
                         ["physics.md", "cooking.md"])
        self.assertEqual(top_relative([]), [])

    def test_select_for_answer_drops_single_term_second_chunk(self):
        chunks = make_chunks()
        # Rank 1 shares "quantum"+"entanglement"; rank 2 shares only "red".
        scored = [
            ScoredChunk(chunks[0], 10.0),
            ScoredChunk(chunks[2], 6.0),
        ]
        selected = select_for_answer(scored, "quantum entanglement red")
        self.assertEqual([s.chunk.source for s in selected], ["physics.md"])

    def test_select_for_answer_keeps_multi_term_second_chunk(self):
        chunks = make_chunks()
        scored = [
            ScoredChunk(chunks[0], 10.0),
            ScoredChunk(chunks[2], 6.0),
        ]
        selected = select_for_answer(scored, "quantum red bicycle")
        self.assertEqual([s.chunk.source for s in selected],
                         ["physics.md", "bikes.md"])

    def test_final_e_stemming_conflates_close_and_closed(self):
        chunks = [Chunk("hours.md", "The shop is closed on Sundays.", 1, 1)]
        retriever = Retriever(chunks)
        results = retriever.search("when does the shop close", k=5)
        self.assertEqual(len(results), 1)

    def test_stemming_matches_word_variants(self):
        chunks = [Chunk("returns.md",
                        "Returns are accepted within thirty days.", 1, 1)]
        retriever = Retriever(chunks)
        # "return" (query) must match "Returns" (document).
        results = retriever.search("how do I return this", k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.source, "returns.md")

    def test_stopwords_do_not_match_on_their_own(self):
        chunks = [Chunk("a.md", "The cat sat on the mat.", 1, 1)]
        retriever = Retriever(chunks)
        self.assertEqual(retriever.search("the a an", k=5), [])
        self.assertEqual(retriever.search("cat", k=5)[0].chunk.source, "a.md")


if __name__ == "__main__":
    unittest.main()
