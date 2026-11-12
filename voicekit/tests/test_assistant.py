import numpy as np

from voicekit import Recognizer, render_stream, train_model
from voicekit.assistant import Assistant
from voicekit.pipeline import Event
from voicekit.wake import WakeGate


class Stub:
    """Recognizer stand-in: releases scripted events once the audio clock passes their end."""
    def __init__(self, events, sr=16000):
        self.events, self.sr, self.t = sorted(events, key=lambda e: e.start_s), sr, 0.0

    def feed(self, chunk):
        self.t += len(chunk) / self.sr
        out = [e for e in self.events if e.end_s <= self.t]
        self.events = [e for e in self.events if e.end_s > self.t]
        return out

    def flush(self):
        out, self.events = self.events, []
        return out


def ev(w, a, b, c=0.95):
    return Event(w, c, a, b)


def run(events, seconds=12, **kw):
    a = Assistant(Stub(events), WakeGate("timer", listen_s=2.0), **kw)
    return a.process(np.zeros(int(seconds * 16000)))


def test_wake_then_command_yields_one_intent():
    cmds = run([ev("timer", 1, 1.5), ev("lights", 2, 2.5), ev("on", 2.8, 3.1)])
    assert len(cmds) == 1 and cmds[0].intent.name == "lights.on" and cmds[0].words == ("lights", "on")


def test_no_wake_no_command():
    assert run([ev("lights", 1, 1.5), ev("on", 1.8, 2.1)]) == []


def test_two_commands_split_by_silence():
    cmds = run([ev("timer", 0, .4), ev("lights", 1, 1.4), ev("on", 1.6, 1.9),
                ev("timer", 8, 8.4), ev("music", 9, 9.4), ev("play", 9.6, 10)])
    assert [c.intent.name for c in cmds] == ["lights.on", "music.play"]


def test_invalid_combo_reported_with_none_intent():
    cmds = run([ev("timer", 0, .4), ev("lights", 1, 1.4), ev("play", 1.6, 1.9)])
    assert len(cmds) == 1 and cmds[0].intent is None and cmds[0].words == ("lights", "play")


def test_command_emitted_only_after_quiet_window_and_flush_emits_pending():
    a = Assistant(Stub([ev("timer", 0, .4), ev("stop", 1, 1.4)]), WakeGate("timer", listen_s=2.0))
    got = []
    for _ in range(int(2.0 / 0.05)):               # 2.0 s of audio: "stop" ended at 1.4, gap 1.2 not yet passed
        got += a.feed(np.zeros(800))
    assert got == []
    assert [c.intent.name for c in a.flush()] == ["stop_all"]


def test_chunk_size_does_not_change_result():
    evs = [ev("timer", 0, .4), ev("weather", 1, 1.5)]
    sig = np.zeros(8 * 16000)
    r = [Assistant(Stub(evs), WakeGate("timer")).process(sig, chunk=c) for c in (400, 800, 3200)]
    assert r[0] == r[1] == r[2] and r[0][0].intent.name == "weather.query"


def test_real_audio_end_to_end():
    model = train_model(50, seed=0, epochs=50)
    rng = np.random.default_rng(7)
    sig, _ = render_stream(["stop", "timer", "lights", "on"], rng, 25, gap=(0.5, 0.7))
    cmds = Assistant(Recognizer(model)).process(sig)
    assert all(c.words[0] != "stop" for c in cmds)
