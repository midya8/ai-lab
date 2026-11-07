"""Tests for ingest (load/save index)."""

import json
import tempfile
import unittest
from pathlib import Path

from ragkit.chunk import Chunk
from ragkit.ingest import load_documents, load_index, save_index


class TestIngest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loads_markdown_and_text_ignores_other_files(self):
        (self.root / "a.md").write_text("Hello world. Goodbye world.")
        (self.root / "b.txt").write_text("Plain text here.")
        (self.root / "c.pdf").write_text("Not a real pdf, should be ignored.")
        (self.root / "d.json").write_text('{"ignored": true}')
        chunks = load_documents(self.root, chunk_size=100, overlap=10)
        sources = {c.source for c in chunks}
        self.assertEqual(sources, {"a.md", "b.txt"})

    def test_missing_directory_raises(self):
        with self.assertRaises(ValueError):
            load_documents(self.root / "nope")

    def test_empty_directory_produces_no_chunks(self):
        self.assertEqual(load_documents(self.root), [])

    def test_empty_file_produces_no_chunks(self):
        (self.root / "empty.md").write_text("")
        self.assertEqual(load_documents(self.root), [])

    def test_save_and_load_round_trip(self):
        chunks = [
            Chunk("a.md", "First chunk.", 1, 2, heading="Intro"),
            Chunk("b.txt", "Second chunk.", 5, 6),
        ]
        index_path = self.root / "index.json"
        save_index(chunks, index_path, chunk_size=800)
        restored = load_index(index_path)
        self.assertEqual(chunks, restored)

    def test_load_missing_index_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_index(self.root / "missing.json")

    def test_load_malformed_index_raises(self):
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"not": "an index"}))
        with self.assertRaises(ValueError):
            load_index(bad)
        bad.write_text("{not json")
        with self.assertRaises(ValueError):
            load_index(bad)


if __name__ == "__main__":
    unittest.main()
