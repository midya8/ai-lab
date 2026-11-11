"""Dataset rendering, the streaming Recognizer, and event scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import synth
from .features import HOP_MS, utterance_vector
from .models import MLP
from .segmenter import Segment, SegmenterConfig, StreamSegmenter, segment_signal
from .synth import SR, VOCAB, OOV_TEST, OOV_TRAIN

WORDS = list(VOCAB)
CLASSES = WORDS + ["_unknown"]
UNKNOWN = len(WORDS)
NOISE_KINDS = ("white", "pink", "hum")


@dataclass
class Event:
    word: str
    confidence: float
    start_s: float
    end_s: float
    probs: dict = field(default_factory=dict, repr=False)


def render_example(label: int, rng: np.random.Generator, snr_db: float, *, kind: str | None = None,
                   wide: bool = False, oov: dict | None = None) -> np.ndarray:
    """One noisy padded utterance of class ``label`` (UNKNOWN = OOV word or non-speech burst)."""
    sp = synth.random_speaker(rng, wide=wide)
    if label == UNKNOWN:
        pool = oov if oov is not None else OOV_TRAIN
        if rng.random() < 0.2:
            w = synth.noise_burst(rng)
        else:
            w = synth.synth_word(list(pool.values())[rng.integers(len(pool))], sp, rng)
    else:
        w = synth.synth_word(VOCAB[WORDS[label]], sp, rng)
    x, region = synth.make_utterance(w, rng)
    return synth.add_noise(x, snr_db, rng, kind or NOISE_KINDS[rng.integers(3)], region=region)


def best_segment(segs: list[Segment]) -> Segment | None:
    segs = [s for s in segs if not s.truncated]
    return max(segs, key=lambda s: len(s.power)) if segs else None


def build_dataset(n_per_class: int, seed: int, snr_range=(0.0, 30.0), *, wide: bool = False,
                  oov: dict | None = None, spectral_sub: bool = True, with_sequences: bool = False,
                  cfg: SegmenterConfig | None = None):
    """Render -> VAD-segment -> featurise. Samples the VAD misses are skipped (counted in ``missed``)."""
    from .features import sequence_features

    rng = np.random.default_rng(seed)
    X, y, snrs, seqs = [], [], [], []
    missed = 0
    for label in range(len(CLASSES)):
        for _ in range(n_per_class):
            snr = float(rng.uniform(*snr_range))
            seg = best_segment(segment_signal(render_example(label, rng, snr, wide=wide, oov=oov), cfg))
            if seg is None:
                missed += 1
                continue
            ns = seg.noise_psd if spectral_sub else None
            X.append(utterance_vector(seg.power, ns))
            if with_sequences:
                seqs.append(sequence_features(seg.power, ns))
            y.append(label)
            snrs.append(snr)
    out = (np.array(X), np.array(y), np.array(snrs), missed)
    return out + (seqs,) if with_sequences else out


def train_model(n_per_class: int = 120, seed: int = 0, **mlp_kwargs) -> MLP:
    X, y, _, _ = build_dataset(n_per_class, seed)
    return MLP(seed=seed, **mlp_kwargs).fit(X, y, n_classes=len(CLASSES))


class Recognizer:
    """Streaming word spotter: audio chunks in, accepted word events out."""

    def __init__(self, model: MLP, threshold: float = 0.6, spectral_sub: bool = True,
                 cfg: SegmenterConfig | None = None):
        self.model, self.threshold, self.spectral_sub = model, threshold, spectral_sub
        self._seg = StreamSegmenter(cfg)
        self.rejected: list[tuple[Segment, str]] = []

    def _classify(self, segs: list[Segment]) -> list[Event]:
        events = []
        for s in segs:
            if s.truncated:
                self.rejected.append((s, "truncated")); continue
            v = utterance_vector(s.power, s.noise_psd if self.spectral_sub else None)
            p = self.model.predict_proba(v)[0]
            k = int(p.argmax())
            if k == UNKNOWN:
                self.rejected.append((s, "unknown")); continue
            if p[k] < self.threshold:
                self.rejected.append((s, "low_confidence")); continue
            events.append(Event(CLASSES[k], float(p[k]), s.start_s, s.end_s, dict(zip(CLASSES, map(float, p)))))
        return events

    def feed(self, chunk: np.ndarray) -> list[Event]:
        return self._classify(self._seg.feed(chunk))

    def flush(self) -> list[Event]:
        return self._classify(self._seg.flush())

    def process(self, x: np.ndarray, chunk: int = 800) -> list[Event]:
        out: list[Event] = []
        for s in range(0, len(x), chunk):
            out += self.feed(x[s:s + chunk])
        return out + self.flush()


def render_stream(words: list[str], rng: np.random.Generator, snr_db: float, *, gap=(0.6, 1.4),
                  lead: float = 1.0, noise_kind: str = "white", tail: float = 1.0):
    """A continuous noisy stream with the given words in order. Returns (signal, truth)
    where truth = [(word, start_s, end_s)]. Noise level is fixed from the nominal speech RMS (0.1)."""
    parts, truth, pos = [np.zeros(int(lead * SR))], [], int(lead * SR)
    for w in words:
        wav = synth.synth_word(VOCAB[w], synth.random_speaker(rng), rng)
        truth.append((w, pos / SR, (pos + len(wav)) / SR))
        parts.append(wav)
        g = np.zeros(int(rng.uniform(*gap) * SR))
        parts.append(g)
        pos += len(wav) + len(g)
    parts.append(np.zeros(int(tail * SR)))
    x = np.concatenate(parts)
    # nominal speech RMS is 0.1 -> compute noise from that, add over the full stream
    ref = np.full(1000, 0.1)
    noise_only = synth.add_noise(np.concatenate([ref, np.zeros(len(x))]), snr_db, rng, noise_kind, region=(0, 1000))
    return x + noise_only[1000:], truth


def score_events(events: list[Event], truth: list[tuple[str, float, float]], min_overlap: float = 0.5) -> dict:
    """Match events to truth words by temporal overlap (>= ``min_overlap`` of the truth span)."""
    used: set[int] = set()
    tp = wrong = 0
    for w, a, b in truth:
        best, best_ov = None, 0.0
        for i, ev in enumerate(events):
            if i in used:
                continue
            ov = max(0.0, min(b, ev.end_s) - max(a, ev.start_s)) / (b - a)
            if ov >= min_overlap and ov > best_ov:
                best, best_ov = i, ov
        if best is not None:
            used.add(best)
            if events[best].word == w:
                tp += 1
            else:
                wrong += 1
    fp = len(events) - tp  # wrong-word + spurious
    fn = len(truth) - tp
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / len(truth) if truth else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "wrong_word": wrong, "precision": prec, "recall": rec, "f1": f1}
