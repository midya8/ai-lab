"""Ingest Markdown / plain-text documents into chunks, and persist an index.

An index is a JSON file: {"chunk_size": ..., "overlap": ...,
"chunks": [chunk dicts]}. It is plain data -- no embeddings, no
server, no network -- so it can be rebuilt any time with ``index``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .chunk import Chunk, split_text

SUPPORTED_SUFFIXES = {".md", ".txt", ".markdown"}


def load_documents(
    docs_dir: str | Path,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Read every supported file under *docs_dir* and chunk it.

    Files are visited in sorted order so indexing is deterministic.
    Empty files produce no chunks. Raises ValueError if *docs_dir*
    is not a directory.
    """
    root = Path(docs_dir)
    if not root.is_dir():
        raise ValueError(f"docs directory not found: {docs_dir}")
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(
            split_text(text, str(path.relative_to(root)), chunk_size, overlap)
        )
    return chunks


def save_index(chunks: list[Chunk], path: str | Path, **meta) -> None:
    """Write chunks (plus indexing parameters) to *path* as JSON."""
    payload = {"chunks": [c.to_dict() for c in chunks], **meta}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_index(path: str | Path) -> list[Chunk]:
    """Read an index file back into chunks.

    Raises FileNotFoundError if the file is missing and ValueError
    if it is not a valid ragkit index.
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read index {path}: {exc}") from exc
    if not isinstance(payload, dict) or "chunks" not in payload:
        raise ValueError(f"not a ragkit index: {path}")
    try:
        return [Chunk.from_dict(item) for item in payload["chunks"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"malformed chunk in index {path}: {exc}") from exc
