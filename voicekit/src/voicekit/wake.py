"""Wake-word gating: only act on commands spoken shortly after a wake word.

``WakeGate`` is a small state machine over time-ordered :class:`Event` objects (it works on the
event timestamps, not wall-clock time, so it behaves identically on live and replayed audio).

* IDLE: every event is dropped except the wake word, which must have confidence >=
  ``wake_confidence`` (a stricter bar than ordinary words, since a false wake is costly).
* ARMED: events starting within ``listen_s`` of the last activity (wake end, or the end of the
  last accepted command word) are passed through. Each accepted word extends the session, so a
  multi-word command such as "lights on" stays in one session. A longer silence returns to IDLE.
* A repeated wake word while armed just refreshes the session and is not forwarded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .pipeline import Event


@dataclass
class WakeGate:
    wake_word: str = "timer"
    listen_s: float = 3.0
    wake_confidence: float = 0.8
    armed_until: float = field(default=-1.0, init=False)
    dropped: list = field(default_factory=list, init=False, repr=False)

    def armed_at(self, t: float) -> bool:
        return t <= self.armed_until

    def push(self, ev: Event) -> Event | None:
        """Feed one event; return it if it should reach the intent parser, else None."""
        if ev.word == self.wake_word:
            if ev.confidence >= self.wake_confidence or self.armed_at(ev.start_s):
                self.armed_until = ev.end_s + self.listen_s
            else:
                self.dropped.append((ev, "weak_wake"))
            return None
        if not self.armed_at(ev.start_s):
            self.dropped.append((ev, "not_armed"))
            return None
        self.armed_until = ev.end_s + self.listen_s
        return ev

    def filter(self, events: list[Event]) -> list[Event]:
        out = []
        for ev in sorted(events, key=lambda e: e.start_s):
            kept = self.push(ev)
            if kept is not None:
                out.append(kept)
        return out
