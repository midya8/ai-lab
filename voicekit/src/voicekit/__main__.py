"""CLI: python -m voicekit {train,demo,segment,synth}"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from . import synth
from .intents import parse_events
from .models import MLP
from .pipeline import Recognizer, train_model
from .segmenter import segment_signal


def _model(path: str | None, per_class: int) -> MLP:
    if path:
        return MLP.load(path)
    print(f"(no --model given: training one on synthetic speech, {per_class}/class ...)", file=sys.stderr)
    return train_model(per_class)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="voicekit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the word classifier on synthetic speech")
    t.add_argument("--out", default="model.npz")
    t.add_argument("--per-class", type=int, default=120)
    t.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("demo", help="synthesize a noisy command stream and recognise it")
    d.add_argument("--model")
    d.add_argument("--snr", type=float, default=10.0)
    d.add_argument("--seed", type=int, default=4)
    d.add_argument("--per-class", type=int, default=150)
    s = sub.add_parser("segment", help="print voice-activity segments of any 16-bit PCM WAV file")
    s.add_argument("wav")
    y = sub.add_parser("synth", help="write a synthetic word to a WAV file")
    y.add_argument("word")
    y.add_argument("--out", default="word.wav")
    y.add_argument("--snr", type=float)
    y.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "train":
        train_model(a.per_class, a.seed).save(a.out)
        print(f"saved {a.out}")
    elif a.cmd == "demo":
        rng = np.random.default_rng(a.seed)
        phrases = [["lights", "on"], ["play", "music"], ["weather"], ["timer", "on"], ["stop"]]
        # short gap inside a phrase, long gap between phrases
        parts, pos, truth, i = [np.zeros(synth.SR)], synth.SR, [], 0
        for p in phrases:
            for j, w in enumerate(p):
                wav = synth.synth_word(synth.VOCAB[w], synth.random_speaker(rng), rng)
                truth.append((w, pos / synth.SR, (pos + len(wav)) / synth.SR))
                gap = np.zeros(int((rng.uniform(0.3, 0.45) if j < len(p) - 1 else 2.5) * synth.SR))
                parts += [wav, gap]
                pos += len(wav) + len(gap)
        x = np.concatenate(parts)
        ref = np.concatenate([np.full(1000, 0.1), np.zeros(len(x))])
        x = x + synth.add_noise(ref, a.snr, rng, "white", region=(0, 1000))[1000:]
        rec = Recognizer(_model(a.model, a.per_class))
        events = rec.process(x)
        print(f"{len(x) / synth.SR:.1f}s of audio at {a.snr:g} dB SNR; said: {' '.join(w for w, _, _ in truth)}")
        for e in events:
            print(f"  {e.start_s:5.2f}-{e.end_s:5.2f}s  {e.word:<8} p={e.confidence:.2f}")
        for group, intent in parse_events(events):
            heard = " ".join(e.word for e in group)
            print(f"  command [{heard}] -> {intent.name if intent else 'not understood'}")
        if rec.rejected:
            print(f"  ({len(rec.rejected)} segments rejected: {[r for _, r in rec.rejected]})")
    elif a.cmd == "segment":
        x = synth.read_wav(a.wav)
        segs = segment_signal(x)
        print(f"{len(x) / synth.SR:.2f}s audio, {len(segs)} segment(s)")
        for g in segs:
            print(f"  {g.start_s:6.2f}-{g.end_s:6.2f}s{'  (truncated)' if g.truncated else ''}")
    elif a.cmd == "synth":
        rng = np.random.default_rng(a.seed)
        if a.word not in synth.VOCAB:
            print(f"unknown word {a.word!r}; choose from {', '.join(synth.VOCAB)}", file=sys.stderr)
            return 2
        w = synth.synth_word(synth.VOCAB[a.word], synth.random_speaker(rng), rng)
        x, region = synth.make_utterance(w, rng)
        if a.snr is not None:
            x = synth.add_noise(x, a.snr, rng, "white", region=region)
        synth.write_wav(a.out, x)
        print(f"wrote {a.out} ({len(x) / synth.SR:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
