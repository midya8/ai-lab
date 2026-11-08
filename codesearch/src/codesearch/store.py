"""Incremental on-disk index: re-parse only files whose content hash changed."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk, chunk_source

STORE_VERSION = 1
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules", "build", "dist",
             ".pytest_cache", ".mypy_cache", ".tox", ".codesearch", "site-packages"}
MAX_FILE_BYTES = 1_000_000


@dataclass
class SyncReport:
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    reused: int = 0
    failed: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        s = f"{len(self.added)} added, {len(self.changed)} changed, {len(self.removed)} removed, {self.reused} reused"
        return s + (f", {len(self.failed)} unparsable" if self.failed else "")


def discover(root: Path, include: list[str] | None = None) -> list[str]:
    """Relative POSIX paths of Python files under ``root`` (sorted)."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
        for fn in sorted(filenames):
            if not fn.endswith(".py"):
                continue
            full = Path(dirpath) / fn
            rel = full.relative_to(root).as_posix()
            if include and not any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in include):
                continue
            try:
                if full.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            found.append(rel)
    return sorted(found)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync(root: Path, store_path: Path | None = None, include: list[str] | None = None,
         parse=chunk_source) -> tuple[list[Chunk], SyncReport]:
    """Bring the cached chunks up to date with the files on disk.

    ``parse`` is injectable so tests can prove unchanged files are not re-parsed.
    """
    root = Path(root)
    cache: dict = {}
    if store_path and Path(store_path).exists():
        try:
            data = json.loads(Path(store_path).read_text())
            if data.get("version") == STORE_VERSION:
                cache = data.get("files", {})
        except (json.JSONDecodeError, OSError):
            cache = {}

    report = SyncReport()
    files: dict[str, dict] = {}
    for rel in discover(root, include):
        raw = (root / rel).read_bytes()
        digest = _sha(raw)
        prev = cache.get(rel)
        if prev and prev.get("sha") == digest:
            files[rel] = prev
            report.reused += 1
            if prev.get("error"):
                report.failed[rel] = prev["error"]
            continue
        try:
            chunks = parse(raw.decode("utf-8", errors="replace"), rel)
        except (SyntaxError, ValueError) as e:
            report.failed[rel] = f"{type(e).__name__}: {e}"
            chunks = []
        files[rel] = {"sha": digest, "chunks": [c.to_dict() for c in chunks]}
        if rel in report.failed:
            files[rel]["error"] = report.failed[rel]
        (report.changed if prev else report.added).append(rel)
    report.removed = sorted(set(cache) - set(files))

    if store_path:
        sp = Path(store_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        tmp = sp.with_suffix(sp.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": STORE_VERSION, "files": files}))
        os.replace(tmp, sp)

    chunks_out = [Chunk.from_dict(d) for rel in sorted(files) for d in files[rel]["chunks"]]
    return chunks_out, report
