"""Polling watcher: keep the on-disk index fresh while files change.

Polling (stat signature scan, then the incremental ``sync``) keeps this
dependency-free and portable. A cycle only calls ``sync`` when the cheap
(path, mtime_ns, size) signature of the tree differs from the previous one,
and ``sync`` itself re-parses only files whose content hash changed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from .store import SyncReport, discover, sync


def signature(root: Path, include: list[str] | None = None) -> dict[str, tuple[int, int]]:
    """Cheap fingerprint of the tree: relative path -> (mtime_ns, size)."""
    sig: dict[str, tuple[int, int]] = {}
    for rel in discover(root, include):
        try:
            st = (Path(root) / rel).stat()
        except OSError:
            continue
        sig[rel] = (st.st_mtime_ns, st.st_size)
    return sig


def watch(root: Path, store: Path, include: list[str] | None = None, interval: float = 1.0,
          on_sync: Callable[[SyncReport, int], None] | None = None,
          max_cycles: int | None = None, sleep=time.sleep) -> int:
    """Sync once, then re-sync whenever the tree changes. Returns the number of syncs run.

    ``max_cycles`` bounds the number of polls (None = until interrupted) and
    ``sleep`` is injectable so tests do not wait.
    """
    syncs = 0
    chunks, report = sync(root, store, include=include)
    syncs += 1
    if on_sync:
        on_sync(report, len(chunks))
    last = signature(root, include)
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        sleep(interval)
        cycles += 1
        now = signature(root, include)
        if now == last:
            continue
        last = now
        chunks, report = sync(root, store, include=include)
        syncs += 1
        if on_sync:
            on_sync(report, len(chunks))
    return syncs
