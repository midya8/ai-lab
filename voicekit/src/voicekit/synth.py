"""Synthetic formant-style "speech" so the whole voice pipeline is testable offline.

Every word is a string of phoneme symbols. Vowels and nasals are voiced: a harmonic
series (pitch f0) whose amplitudes follow two formant resonances that glide between
phonemes. Fricatives are band-shaped noise. ``-`` is a short silence (a stop closure).

This is NOT real speech. It has the properties that matter for engineering a voice
pipeline (speaker pitch / vocal-tract / rate variation, additive noise, silence between
words) and none of the messiness of a microphone. Results on it say nothing about
accuracy on human voices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 8000

# (F1, F2) in Hz for a neutral vocal tract
VOWELS = {"a": (730, 1090), "i": (270, 2290), "u": (300, 870), "e": (530, 1840), "o": (570, 840)}
NASALS = {"m": (250, 1100), "n": (250, 1600)}
# (centre Hz, bandwidth Hz)
FRICS = {"s": (3200, 1100), "f": (2300, 2200), "h": (1700, 3000)}
SILENCE = "-"
_BASE_MS = {"vowel": 110, "nasal": 80, "fric": 100, "sil": 50}

# The command vocabulary (pseudo-pronunciations).
VOCAB = {
    "on": "on",
    "off": "of",
    "lights": "naiis",
    "music": "muisi",
    "timer": "saime",
    "stop": "s-ou",
    "play": "-eei",
    "weather": "uese",
}
# Out-of-vocabulary words: some used to train the "unknown" class, others held out.
OOV_TRAIN = {"banana": "manana", "window": "uinuo", "coffee": "ofi", "hello": "heeo"}
OOV_TEST = {"garage": "aa-as", "yellow": "ieno", "tomato": "mameo", "violin": "fiioni"}


@dataclass(frozen=True)
class Speaker:
    f0: float  # Hz, fundamental
    tract: float  # formant scale (vocal tract length proxy)
    rate: float  # speaking rate multiplier


def random_speaker(rng: np.random.Generator, wide: bool = False) -> Speaker:
    """Training-distribution speaker; ``wide=True`` draws vocal tracts outside the training range."""
    if wide:
        tract = rng.uniform(0.78, 0.90) if rng.random() < 0.5 else rng.uniform(1.14, 1.28)
        return Speaker(float(rng.uniform(80, 260)), float(tract), float(rng.uniform(0.8, 1.25)))
    return Speaker(float(rng.uniform(90, 230)), float(rng.uniform(0.92, 1.12)), float(rng.uniform(0.85, 1.2)))


def _kind(ch: str) -> str:
    if ch in VOWELS:
        return "vowel"
    if ch in NASALS:
        return "nasal"
    if ch in FRICS:
        return "fric"
    if ch == SILENCE:
        return "sil"
    raise ValueError(f"unknown phoneme {ch!r}")


def _shaped_noise(n: int, centre: float, bw: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / sr)
    spec *= np.exp(-0.5 * ((f - centre) / (bw / 2.0)) ** 2)
    out = np.fft.irfft(spec, n)
    return out / (np.sqrt(np.mean(out**2)) + 1e-12)


def synth_word(pron: str, speaker: Speaker, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Synthesize one word (no padding), normalized to RMS 0.1 over its active samples."""
    if not pron:
        raise ValueError("empty pronunciation")
    kinds = [_kind(c) for c in pron]
    durs = np.array(
        [max(int(_BASE_MS[k] * 1e-3 * sr / speaker.rate * rng.uniform(0.85, 1.15)), 16) for k in kinds]
    )
    n = int(durs.sum())
    centres = np.cumsum(durs) - durs / 2.0

    # formant / amplitude control points; non-voiced phonemes inherit neighbouring formants
    f1p, f2p, avp = [], [], []
    last = None
    for c, k in zip(pron, kinds):
        if k in ("vowel", "nasal"):
            f1, f2 = (VOWELS if k == "vowel" else NASALS)[c]
            last = (f1 * speaker.tract, f2 * speaker.tract)
        elif last is None:  # leading fricative/silence: borrow the first voiced phoneme
            first = next(x for x in pron if x in VOWELS or x in NASALS)
            f1, f2 = (VOWELS.get(first) or NASALS[first])
            last = (f1 * speaker.tract, f2 * speaker.tract)
        f1p.append(last[0]); f2p.append(last[1])
        avp.append({"vowel": 1.0, "nasal": 0.35}.get(k, 0.0))
    t = np.arange(n)
    F1 = np.interp(t, centres, f1p)
    F2 = np.interp(t, centres, f2p)
    Av = np.interp(t, centres, avp)

    decl = 1.1 - 0.2 * t / n
    vib = 1.0 + 0.02 * np.sin(2 * np.pi * 5.0 * t / sr + rng.uniform(0, 6.28))
    f0 = speaker.f0 * decl * vib
    phase = 2 * np.pi * np.cumsum(f0) / sr
    n_h = int(3800 / f0.min())
    h = np.arange(1, n_h + 1)[:, None]
    freq = h * f0[None, :]
    amp = 1.0 / (1 + ((freq - F1) / 70.0) ** 2) + 0.6 / (1 + ((freq - F2) / 110.0) ** 2)
    amp = amp * (freq < 3800) / np.sqrt(h)
    voiced = (amp * np.sin(h * phase[None, :])).sum(0) * Av
    peak = np.abs(voiced).max()
    if peak > 0:
        voiced = voiced / peak

    noise = np.zeros(n)
    pos = 0
    for c, k, d in zip(pron, kinds, durs):
        if k == "fric":
            centre, bw = FRICS[c]
            seg = _shaped_noise(int(d), centre, bw, sr, rng) * 0.3
            ramp = min(int(0.02 * sr), int(d) // 2)
            env = np.ones(int(d)); env[:ramp] = np.linspace(0, 1, ramp); env[-ramp:] = np.linspace(1, 0, ramp)
            noise[pos:pos + int(d)] = seg * env
        pos += int(d)

    x = voiced + noise
    ramp = int(0.015 * sr)
    x[:ramp] *= np.linspace(0, 1, ramp)
    x[-ramp:] *= np.linspace(1, 0, ramp)
    return _normalize_active(x)


def _normalize_active(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    active = np.abs(x) > 0.02 * np.abs(x).max()
    rms = np.sqrt(np.mean(x[active] ** 2)) + 1e-12
    return x * (target_rms / rms)


def noise_burst(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """A non-speech transient (cough / door / click-like): short band-limited noise."""
    n = int(rng.uniform(0.15, 0.4) * sr)
    centre = rng.uniform(300, 3200)
    x = _shaped_noise(n, centre, rng.uniform(300, 1800), sr, rng)
    x *= np.hanning(n)
    return _normalize_active(x)


def make_utterance(
    word: np.ndarray, rng: np.random.Generator, pad_range=(0.3, 0.6), sr: int = SR
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a word with silence on both sides. Returns (signal, (start, end) sample indexes)."""
    a = int(rng.uniform(*pad_range) * sr)
    b = int(rng.uniform(*pad_range) * sr)
    return np.concatenate([np.zeros(a), word, np.zeros(b)]), (a, a + len(word))


def add_noise(
    x: np.ndarray, snr_db: float, rng: np.random.Generator, kind: str = "white",
    region: tuple[int, int] | None = None,
) -> np.ndarray:
    """Add noise so that speech power (over ``region``, default: active samples) / noise power = SNR."""
    if region is not None:
        seg = x[region[0]:region[1]]
    else:
        seg = x[np.abs(x) > 0.02 * np.abs(x).max()]
    p_sig = float(np.mean(seg**2))
    n = len(x)
    w = rng.standard_normal(n)
    if kind in ("pink", "hum"):
        spec = np.fft.rfft(w)
        f = np.fft.rfftfreq(n, 1.0 / SR)
        f[0] = f[1]
        w = np.fft.irfft(spec / np.sqrt(f / f[1]), n)
    elif kind != "white":
        raise ValueError(f"unknown noise kind {kind!r}")
    w = w / (np.sqrt(np.mean(w**2)) + 1e-12)
    if kind == "hum":
        tt = np.arange(n) / SR
        hum = sum(a * np.sin(2 * np.pi * f0 * tt + rng.uniform(0, 6.28)) for f0, a in ((60, 1.0), (120, 0.6), (180, 0.4)))
        w = w + 0.7 * hum / np.sqrt(np.mean(hum**2))
        w = w / np.sqrt(np.mean(w**2))
    sigma = np.sqrt(p_sig / (10 ** (snr_db / 10.0)))
    return x + sigma * w


def write_wav(path: str, x: np.ndarray, sr: int = SR) -> None:
    import wave

    y = np.clip(x / (np.abs(x).max() + 1e-12) * 0.9, -1, 1)
    with wave.open(path, "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((y * 32767).astype("<i2").tobytes())


def read_wav(path: str, sr: int = SR) -> np.ndarray:
    """Read a 16-bit PCM WAV, mix to mono and resample (crude boxcar low-pass + linear interp) to ``sr``."""
    import wave

    with wave.open(path, "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        ch, src = f.getnchannels(), f.getframerate()
        raw = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    x = raw.reshape(-1, ch).mean(1)
    if src != sr:
        k = max(int(round(src / sr)), 1)
        if k > 1:
            x = np.convolve(x, np.ones(k) / k, mode="same")
        t_new = np.arange(int(len(x) * sr / src)) * (src / sr)
        x = np.interp(t_new, np.arange(len(x)), x)
    return x
