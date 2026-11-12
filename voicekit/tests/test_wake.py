import numpy as np

from voicekit import Recognizer, parse_events, render_stream, train_model
from voicekit.pipeline import Event
from voicekit.wake import WakeGate


def ev(w, a, b, c=0.95):
    return Event(w, c, a, b)


def test_commands_before_wake_are_dropped():
    g = WakeGate("timer")
    out = g.filter([ev("lights", 0, .5), ev("on", .6, 1.0)])
    assert out == [] and [r for _, r in g.dropped] == ["not_armed"] * 2


def test_wake_then_command_passes_and_wake_not_forwarded():
    out = WakeGate("timer").filter([ev("timer", 0, .5), ev("lights", 1, 1.5), ev("on", 1.7, 2.1)])
    assert [e.word for e in out] == ["lights", "on"]


def test_session_expires_after_silence():
    out = WakeGate("timer", listen_s=2).filter([ev("timer", 0, .5), ev("lights", 1, 1.5), ev("on", 6, 6.4)])
    assert [e.word for e in out] == ["lights"]


def test_accepted_word_extends_session():
    out = WakeGate("timer", listen_s=2).filter([ev("timer", 0, .5), ev("lights", 2, 2.5), ev("on", 4, 4.4)])
    assert [e.word for e in out] == ["lights", "on"]


def test_weak_wake_does_not_arm_but_repeat_wake_refreshes():
    g = WakeGate("timer", listen_s=2, wake_confidence=0.8)
    assert g.filter([ev("timer", 0, .5, 0.6), ev("lights", 1, 1.4)]) == []
    assert g.dropped[0][1] == "weak_wake"
    g2 = WakeGate("timer", listen_s=2, wake_confidence=0.8)
    out = g2.filter([ev("timer", 0, .5), ev("timer", 2, 2.5, 0.5), ev("music", 4, 4.5)])
    assert [e.word for e in out] == ["music"]


def test_end_to_end_stream_with_intent_parsing():
    model = train_model(50, seed=0, epochs=50)
    rng = np.random.default_rng(7)
    # "stop" is spoken with no wake word first, so it must be ignored
    sig, _ = render_stream(["stop", "timer", "lights", "on"], rng, 25, gap=(0.5, 0.7))
    events = Recognizer(model).process(sig)
    kept = WakeGate("timer").filter(events)
    assert "stop" not in [e.word for e in kept]
    parsed = parse_events(kept)
    assert all(it is None or it.words[0] in {"lights", "on"} for _, it in parsed)
