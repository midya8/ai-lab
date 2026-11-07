"""Tests for the extractive (offline) answerer."""

import unittest

from ragkit.answer import Citation, ExtractiveAnswerer
from ragkit.chunk import Chunk


def chunk(source, text, start=1, end=3):
    return Chunk(source, text, start, end)


class TestExtractiveAnswerer(unittest.TestCase):
    def test_answer_cites_sources_with_line_ranges(self):
        answerer = ExtractiveAnswerer()
        chunks = [
            chunk("returns.md", "Returns are accepted within 30 days of purchase.", 4, 6),
        ]
        answer = answerer.answer("How long do I have to return a bike?", chunks)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0], Citation("returns.md", 4, 6))
        self.assertIn("[returns.md:4-6]", answer.text)

    def test_answer_quotes_chunk_text(self):
        answerer = ExtractiveAnswerer()
        chunks = [chunk("repairs.md", "A standard tune-up costs $89.", 4, 5)]
        answer = answerer.answer("What does a tune-up cost?", chunks)
        self.assertIn("$89", answer.text)

    def test_no_chunks_gives_no_result_answer(self):
        answerer = ExtractiveAnswerer()
        answer = answerer.answer("anything?", [])
        self.assertEqual(answer.citations, [])
        self.assertIn("couldn't find anything relevant", answer.text)

    def test_max_chunks_is_respected(self):
        answerer = ExtractiveAnswerer(max_chunks=2)
        chunks = [
            chunk(f"doc{i}.md", f"Sentence number {i} here.", 1, 1)
            for i in range(5)
        ]
        answer = answerer.answer("sentence", chunks)
        self.assertEqual(len(answer.citations), 2)

    def test_deterministic_same_input_same_answer(self):
        answerer = ExtractiveAnswerer()
        chunks = [chunk("a.md", "The quick brown fox jumps.", 1, 1)]
        first = answerer.answer("fox", chunks)
        second = answerer.answer("fox", chunks)
        self.assertEqual(first.text, second.text)
        self.assertEqual(first.citations, second.citations)

    def test_long_chunk_excerpt_stops_at_sentence_boundary(self):
        answerer = ExtractiveAnswerer(max_chars=40)
        text = "First short sentence here. Second sentence follows here. Third one."
        answer = answerer.answer("sentence", [chunk("a.md", text, 1, 3)])
        # Excerpt must not end mid-sentence.
        self.assertTrue(
            answer.text.rstrip().endswith((".", "!", "?")),
            answer.text,
        )


if __name__ == "__main__":
    unittest.main()
