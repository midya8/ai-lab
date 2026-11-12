"""End-to-end voice assistant: audio chunks -> Recognizer -> WakeGate -> intent grammar.

``Assistant.feed`` accepts audio chunks of any size and returns the :class:`Command` objects that
became complete while that audio was consumed. A command is complete once the wake session has
been quiet for ``group_gap_s`` (so "lights ... on" is parsed as one command), which means a
command is emitted slightly after it is spoken; ``flush`` emits whatever is pending at end of
stream. Everything is driven by event timestamps, so live and replayed audio behave identically.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intents import Intent, parse_words
from .pipeline import Event, Recognizer
from .wake import WakeGate


@dataclass(frozen=True)
class Command:
    intent: Intent | None   # None = words were heard after the wake word but are not a valid command
    words: tuple[str, ...]
    start_s: float
    end_s: float


class Assistant:
    def __init__(self, recognizer: Recognizer, gate: WakeGate | None = None, group_gap_s: float = 1.2,
                 sample_rate: int = 16000):
        self.recognizer = recognizer
        self.gate = gate or WakeGate()
        self.group_gap_s = group_gap_s
        self.sample_rate = sample_rate
        self._pending: list[Event] = []
        self._seen_s = 0.0

    def _close(self) -> Command | None:
        if not self._pending:
            return None
        evs, self._pending = self._pending, []
        words = tuple(e.word for e in evs)
        return Command(parse_words(words), words, evs[0].start_s, evs[-1].end_s)

    def _ingest(self, events: list[Event]) -> list[Command]:
        out: list[Command] = []
        for ev in sorted(events, key=lambda e: e.start_s):
            if self._pending and ev.start_s - self._pending[-1].end_s > self.group_gap_s:
                out.append(self._close())
            kept = self.gate.push(ev)
            if kept is not None:
                self._pending.append(kept)
        return out

    def _expire(self) -> list[Command]:
        """Close a pending command once the audio clock is past its quiet window."""
        if self._pending and self._seen_s - self._pending[-1].end_s > self.group_gap_s:
            return [self._close()]
        return []

    def feed(self, chunk: np.ndarray) -> list[Command]:
        self._seen_s += len(chunk) / self.sample_rate
        out = self._ingest(self.recognizer.feed(chunk))
        return out + self._expire()

    def flush(self) -> list[Command]:
        out = self._ingest(self.recognizer.flush())
        last = self._close()
        return out + ([last] if last else [])

    def process(self, x: np.ndarray, chunk: int = 800) -> list[Command]:
        out: list[Command] = []
        for i in range(0, len(x), chunk):
            out += self.feed(x[i:i + chunk])
        return out + self.flush()
