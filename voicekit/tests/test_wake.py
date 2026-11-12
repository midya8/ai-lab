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


def test_merge_duplicates_joins_split_word():
    from voicekit.wake import merge_duplicates
    out = merge_duplicates([ev("on", 1.0, 1.2, .7), ev("on", 1.25, 1.5, .9)])
    assert len(out) == 1 and out[0].start_s == 1.0 and out[0].end_s == 1.5 and out[0].confidence == .9


def test_merge_duplicates_keeps_distinct_or_distant():
    from voicekit.wake import merge_duplicates
    evs = [ev("on", 1.0, 1.2), ev("off", 1.25, 1.5), ev("on", 2.0, 2.2)]
    assert merge_duplicates(evs) == evs
    assert len(merge_duplicates([ev("on", 0, .2), ev("on", .5, .7)], gap_s=0.1)) == 2


def test_merge_duplicates_unsorted_and_overlap():
    from voicekit.wake import merge_duplicates
    out = merge_duplicates([ev("on", 1.1, 1.4), ev("on", 1.0, 1.3)])
    assert len(out) == 1 and out[0].end_s == 1.4


def test_margin_filter_drops_ambiguous_keeps_probless():
    from voicekit.wake import margin, margin_filter
    clear = Event("on", .9, 0, .4, {"on": .9, "off": .05, "x": .05})
    vague = Event("off", .45, .5, .9, {"off": .45, "on": .40, "x": .15})
    bare = Event("up", .6, 1, 1.4)
    assert abs(margin(clear) - .85) < 1e-9 and margin(bare) is None
    assert margin_filter([clear, vague, bare], 0.2) == [clear, bare]
    assert margin_filter([vague], 0.0) == [vague]


def test_max_session_caps_continuous_speech():
    evs = [ev("timer", 0, .5)] + [ev("on", 1 + i, 1.4 + i) for i in range(10)]
    uncapped = WakeGate("timer", listen_s=2).filter(evs)
    capped = WakeGate("timer", listen_s=2, max_session_s=4).filter(evs)
    assert len(uncapped) == 10
    assert [e.start_s for e in capped] == [1, 2, 3, 4]


def test_new_wake_after_cap_starts_fresh_session():
    g = WakeGate("timer", listen_s=2, max_session_s=3)
    out = g.filter([ev("timer", 0, .5), ev("on", 1, 1.4), ev("on", 5, 5.4), ev("timer", 6, 6.5), ev("off", 7, 7.4)])
    assert [e.start_s for e in out] == [1, 7]
