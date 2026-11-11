"""Turn recognised words into device commands with a tiny order-tolerant grammar."""
from __future__ import annotations

from dataclasses import dataclass

DEVICES = {"lights", "music", "timer", "weather"}
ACTIONS = {"on", "off", "play", "stop"}


@dataclass(frozen=True)
class Intent:
    name: str
    words: tuple[str, ...]


def group_events(events, window_s: float = 1.2) -> list[list]:
    """Group time-ordered events whose gap to the previous one is <= ``window_s``."""
    groups: list[list] = []
    for ev in sorted(events, key=lambda e: e.start_s):
        if groups and ev.start_s - groups[-1][-1].end_s <= window_s:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    return groups


def parse_words(words) -> Intent | None:
    """Words -> Intent, or None if the combination is not a valid command."""
    ws: list[str] = []
    for w in words:  # collapse immediate repeats (a word heard twice)
        if not ws or ws[-1] != w:
            ws.append(w)
    key = tuple(ws)
    devices = [w for w in ws if w in DEVICES]
    actions = [w for w in ws if w in ACTIONS]
    if len(devices) + len(actions) != len(ws) or len(devices) > 1 or len(actions) > 1:
        return None
    dev = devices[0] if devices else None
    act = actions[0] if actions else None
    if dev == "weather" and act is None:
        return Intent("weather.query", key)
    if dev in ("lights", "timer") and act in ("on", "off"):
        return Intent(f"{dev}.{act}", key)
    if dev == "music" and act in ("play", "stop"):
        return Intent(f"music.{act}", key)
    if dev is None and act == "play":
        return Intent("music.play", key)
    if dev is None and act == "stop":
        return Intent("stop_all", key)
    return None


def parse_events(events, window_s: float = 1.2) -> list[tuple[list, Intent | None]]:
    return [(g, parse_words([e.word for e in g])) for g in group_events(events, window_s)]
