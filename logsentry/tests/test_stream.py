import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from logsentry import parse_line, detect, generate
from logsentry.stream import StreamDetector


def evs():
    return [e for e in map(parse_line, generate()) if e]


def run_stream(events, **kw):
    sd, out = StreamDetector(**kw), []
    for e in events:
        out += sd.feed(e)
    return out + sd.flush()


def key(a):
    return (a.kind, a.window, a.score, a.detail)


def test_stream_matches_batch():
    events = evs()
    batch = sorted(map(key, detect(events)))
    assert batch  # demo has incidents
    assert sorted(map(key, run_stream(events))) == batch


def test_incremental_emission_before_end():
    sd, first_at = StreamDetector(), None
    events = evs()
    for i, e in enumerate(events):
        if sd.feed(e) and first_at is None:
            first_at = i
    assert first_at is not None and first_at < len(events) - 1


def test_bounded_history_and_order_check():
    sd = StreamDetector(history=10)
    for e in evs():
        sd.feed(e)
    assert len(sd.volume) <= 10
    with pytest.raises(ValueError):
        sd.feed(evs()[0])
