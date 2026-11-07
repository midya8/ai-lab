"""Tests for sentence-aware chunking."""

import unittest

from ragkit.chunk import Chunk, split_text

SENTENCES = (
    "The shop opens at nine in the morning. "
    "Repairs are booked online or in store. "
    "A standard tune-up costs eighty-nine dollars. "
    "Loaner bikes are available for fifteen dollars per day. "
    "The shop is closed on Sundays so the team can ride."
)


class TestSplitText(unittest.TestCase):
    def test_empty_text_produces_no_chunks(self):
        self.assertEqual(split_text("", "a.md"), [])
        self.assertEqual(split_text("   \n  ", "a.md"), [])

    def test_single_sentence_is_one_chunk(self):
        chunks = split_text("Hello world.", "a.md")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "Hello world.")
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (1, 1))

    def test_chunks_do_not_cut_mid_sentence(self):
        chunks = split_text(SENTENCES, "a.md", chunk_size=100, overlap=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Every chunk ends at a sentence boundary (or is one long
            # sentence that exceeded the chunk size on its own).
            self.assertTrue(
                chunk.text.endswith((".", "!", "?")),
                f"chunk cut mid-sentence: {chunk.text!r}",
            )

    def test_overlap_shares_sentences_between_chunks(self):
        chunks = split_text(SENTENCES, "a.md", chunk_size=100, overlap=60)
        self.assertGreater(len(chunks), 1)
        for prev, nxt in zip(chunks, chunks[1:]):
            prev_sentences = {s.strip() for s in prev.text.split(".") if s.strip()}
            next_sentences = {s.strip() for s in nxt.text.split(".") if s.strip()}
            self.assertTrue(
                prev_sentences & next_sentences,
                f"no shared sentence between {prev.text!r} and {nxt.text!r}",
            )

    def test_overlap_is_bounded_by_chunk_size(self):
        chunks = split_text(SENTENCES, "a.md", chunk_size=100, overlap=60)
        for chunk in chunks:
            # Overlap sentences may push a chunk slightly over, but never
            # by more than the overlap itself plus one sentence.
            self.assertLessEqual(len(chunk.text), 100 + 60 + 60)

    def test_line_ranges_track_source_lines(self):
        text = (
            "First line ends here.\n"
            "Second line ends here.\n"
            "Third line ends here.\n"
            "Fourth line ends here.\n"
            "Fifth line ends here.\n"
        )
        chunks = split_text(text, "a.md", chunk_size=45, overlap=0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].start_line, 1)
        # Chunks are contiguous in line space.
        for prev, nxt in zip(chunks, chunks[1:]):
            self.assertLessEqual(prev.end_line, nxt.start_line)
            self.assertEqual(nxt.start_line, prev.start_line + 1
                             if prev.end_line == prev.start_line else nxt.start_line)

    def test_heading_context_is_recorded(self):
        text = (
            "# Returns\n\n"
            "Returns are accepted within thirty days.\n\n"
            "## Exchanges\n\n"
            "Exchanges are free and done in store.\n"
        )
        chunks = split_text(text, "a.md", chunk_size=60, overlap=0)
        headings = {c.heading for c in chunks}
        self.assertIn("Returns", headings)
        self.assertIn("Exchanges", headings)

    def test_long_sentence_becomes_its_own_chunk(self):
        long_sentence = "Word " * 200 + "."
        chunks = split_text(long_sentence.strip(), "a.md", chunk_size=100, overlap=10)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].text.startswith("Word"))

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            split_text("Hello.", "a.md", chunk_size=0)
        with self.assertRaises(ValueError):
            split_text("Hello.", "a.md", chunk_size=100, overlap=100)
        with self.assertRaises(ValueError):
            split_text("Hello.", "a.md", chunk_size=100, overlap=-1)

    def test_citation_format(self):
        chunk = Chunk(source="returns.md", text="x", start_line=12, end_line=24)
        self.assertEqual(chunk.citation(), "[returns.md:12-24]")

    def test_chunk_round_trips_through_dict(self):
        chunk = Chunk("a.md", "Some text.", 3, 5, heading="Intro")
        restored = Chunk.from_dict(chunk.to_dict())
        self.assertEqual(chunk, restored)


if __name__ == "__main__":
    unittest.main()
