"""Streaming voice-activity detection / utterance segmentation.

Chunk-size invariant: feeding a signal in any chunking yields identical segments. An adaptive
noise floor (EMA of band energy and of the power spectrum) is updated only while idle, so
speech does not pollute it; each segment carries a snapshot of the noise spectrum for
spectral subtraction. Limitations: the first ``warmup`` frames must be noise-only, and a
sudden, sustained rise in background level is reported as a truncated segment and then
absorbed into the floor.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import _FB, HOP_MS, frame_len, frame_power, hop_len
from .synth import SR


@dataclass(frozen=True)
class SegmenterConfig:
    on_margin: float = 0.95     # activity score (see StreamSegmenter._score) needed to trigger
    off_margin: float = 0.70    # hysteresis: stay active above this
    top_bands: int = 4          # mel bands averaged in the score
    start_frames: int = 3       # consecutive frames above on-threshold to open
    hangover_frames: int = 15   # quiet frames tolerated inside/after a word
    pre_frames: int = 3         # pre-roll kept before the trigger
    tail_frames: int = 2
    min_frames: int = 10        # voiced spans shorter than this are dropped (100 ms)
    max_frames: int = 150       # longer segments are cut and flagged truncated
    warmup_frames: int = 15
    noise_alpha: float = 0.05


@dataclass
class Segment:
    start_frame: int
    end_frame: int  # exclusive
    power: np.ndarray  # (T, bins)
    noise_psd: np.ndarray
    truncated: bool = False

    @property
    def start_s(self) -> float:
        return self.start_frame * HOP_MS / 1000.0

    @property
    def end_s(self) -> float:
        return self.end_frame * HOP_MS / 1000.0


class StreamSegmenter:
    def __init__(self, cfg: SegmenterConfig | None = None, sr: int = SR):
        self.cfg = cfg or SegmenterConfig()
        self.sr = sr
        self._fl, self._hl = frame_len(sr), hop_len(sr)
        self._buf = np.zeros(0)
        self._i = 0  # next frame index
        self._warm_P: list[np.ndarray] = []
        self._noise_psd: np.ndarray | None = None
        self._noise_mel: np.ndarray | None = None
        self._recent: deque = deque(maxlen=self.cfg.pre_frames + self.cfg.start_frames)
        self._cnt_on = 0
        self._active = False
        self._frames: list[tuple[int, np.ndarray]] = []
        self._first_voiced = -1
        self._last_voiced = -1
        self._snapshot: np.ndarray | None = None

    # -- public -------------------------------------------------------------------------
    def feed(self, chunk: np.ndarray) -> list[Segment]:
        self._buf = np.concatenate([self._buf, np.asarray(chunk, dtype=np.float64)])
        out: list[Segment] = []
        while len(self._buf) >= self._fl:
            out.extend(self._frame(frame_power(self._buf[: self._fl])))
            self._buf = self._buf[self._hl:]
        return out

    def flush(self) -> list[Segment]:
        out: list[Segment] = []
        if self._active:
            seg = self._close(truncated=False)
            if seg is not None:
                out.append(seg)
        return out

    # -- internals ----------------------------------------------------------------------
    def _score(self, P: np.ndarray) -> float:
        """Mean of the ``top_bands`` largest per-mel-band log10 excesses over the noise floor.

        Speech concentrates its energy in a few formant regions, so the strongest bands
        stand out at SNRs where total band energy barely moves; noise-only frames stay low
        because each mel band averages many FFT bins.
        """
        r = np.log10((P @ _FB.T + 1e-10) / self._noise_mel)
        return float(np.sort(r)[-self.cfg.top_bands:].mean())

    def _set_noise(self, psd: np.ndarray) -> None:
        self._noise_psd = psd
        self._noise_mel = psd @ _FB.T + 1e-10

    def _frame(self, P: np.ndarray) -> list[Segment]:
        c, i = self.cfg, self._i
        self._i += 1
        if i < c.warmup_frames:
            self._warm_P.append(P)
            if i == c.warmup_frames - 1:
                self._set_noise(np.mean(self._warm_P, axis=0))
            return []
        e = self._score(P)
        if not self._active:
            self._recent.append((i, P))
            if e > c.on_margin:
                self._cnt_on += 1
            else:
                self._cnt_on = 0
                a = c.noise_alpha
                self._set_noise((1 - a) * self._noise_psd + a * P)
            if self._cnt_on >= c.start_frames:
                self._active = True
                self._frames = list(self._recent)
                self._first_voiced = i - c.start_frames + 1
                self._last_voiced = i
                self._snapshot = self._noise_psd.copy()
            return []
        self._frames.append((i, P))
        if e > c.off_margin:
            self._last_voiced = i
        if i - self._last_voiced >= c.hangover_frames:
            seg = self._close(truncated=False)
            return [seg] if seg is not None else []
        if len(self._frames) >= c.max_frames:
            seg = self._close(truncated=True)
            # sustained: treat the tail as the new background level
            tail = np.mean([p for _, p in self._frames_before_close[-10:]], axis=0)
            self._set_noise(tail)
            return [seg] if seg is not None else []
        return []

    def _close(self, truncated: bool) -> Segment | None:
        c = self.cfg
        frames = self._frames
        self._frames_before_close = frames
        self._active = False
        self._cnt_on = 0
        self._recent.clear()
        self._frames = []
        if not truncated:
            keep_until = self._last_voiced + c.tail_frames
            frames = [(i, p) for i, p in frames if i <= keep_until]
        if self._last_voiced - self._first_voiced + 1 < c.min_frames and not truncated:
            return None
        return Segment(
            start_frame=frames[0][0],
            end_frame=frames[-1][0] + 1,
            power=np.stack([p for _, p in frames]),
            noise_psd=self._snapshot,
            truncated=truncated,
        )


def segment_signal(x: np.ndarray, cfg: SegmenterConfig | None = None, sr: int = SR) -> list[Segment]:
    """Offline convenience: run the streaming segmenter over a whole signal."""
    s = StreamSegmenter(cfg, sr)
    return s.feed(x) + s.flush()
