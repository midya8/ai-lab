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


def merge_duplicates(events: list[Event], gap_s: float = 0.15) -> list[Event]:
    """Collapse back-to-back detections of the same word into one event.

    A word split by the segmenter (e.g. at a plosive closure) can be reported twice. Consecutive
    events with the same word whose gap is <= ``gap_s`` (or that overlap) are merged: the span is
    the union and the confidence is the maximum.
    """
    out: list[Event] = []
    for ev in sorted(events, key=lambda e: e.start_s):
        prev = out[-1] if out else None
        if prev is not None and prev.word == ev.word and ev.start_s - prev.end_s <= gap_s:
            out[-1] = Event(prev.word, max(prev.confidence, ev.confidence), prev.start_s, max(prev.end_s, ev.end_s))
        else:
            out.append(ev)
    return out


def margin(ev: Event) -> float | None:
    """Gap between the top two class probabilities (None when the event carries no ``probs``)."""
    vals = sorted(ev.probs.values(), reverse=True) if ev.probs else []
    if len(vals) < 2:
        return None
    return float(vals[0] - vals[1])


def margin_filter(events: list[Event], min_margin: float = 0.2) -> list[Event]:
    """Drop events whose top-2 probability margin is below ``min_margin`` (ambiguous words).

    Events without ``probs`` are kept, since there is nothing to judge them by.
    """
    out = []
    for ev in events:
        m = margin(ev)
        if m is None or m >= min_margin:
            out.append(ev)
    return out
