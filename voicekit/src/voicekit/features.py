"""Log-mel / MFCC features from scratch (NumPy), with spectral subtraction."""
from __future__ import annotations

import numpy as np

from .synth import SR

FRAME_MS, HOP_MS, N_FFT, N_MELS = 25, 10, 256, 24


def frame_len(sr: int = SR) -> int:
    return int(FRAME_MS * sr / 1000)


def hop_len(sr: int = SR) -> int:
    return int(HOP_MS * sr / 1000)


def hz2mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)


def mel2hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SR, fmin: float = 80.0,
                   fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2.0
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, mid, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (bins - lo) / (mid - lo)
        down = (hi - bins) / (hi - mid)
        fb[i] = np.maximum(0.0, np.minimum(up, down))
    return fb


_WINDOW = np.hamming(frame_len())


def frame_power(frame: np.ndarray) -> np.ndarray:
    """Power spectrum of one frame (per-frame pre-emphasis, Hamming window)."""
    f = frame - 0.97 * np.concatenate([[frame[0]], frame[:-1]])
    spec = np.fft.rfft(f * _WINDOW, N_FFT)
    return spec.real**2 + spec.imag**2


def power_frames(x: np.ndarray) -> np.ndarray:
    """All frames of a signal -> (T, N_FFT/2+1) power spectra."""
    fl, hl = frame_len(), hop_len()
    if len(x) < fl:
        return np.zeros((0, N_FFT // 2 + 1))
    n = 1 + (len(x) - fl) // hl
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - 0.97 * np.concatenate([fr[:, :1], fr[:, :-1]], axis=1)
    spec = np.fft.rfft(fr * _WINDOW[None, :], N_FFT, axis=1)
    return spec.real**2 + spec.imag**2


def spectral_subtract(P: np.ndarray, noise_psd: np.ndarray, over: float = 1.5, floor: float = 0.05) -> np.ndarray:
    """Power-domain spectral subtraction with a spectral floor (limits 'musical noise')."""
    return np.maximum(P - over * noise_psd[None, :], floor * P)


def _dct_matrix(n_mfcc: int, n_mels: int) -> np.ndarray:
    k = np.arange(n_mfcc)[:, None]
    n = np.arange(n_mels)[None, :]
    m = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels)) * np.sqrt(2.0 / n_mels)
    m[0] *= 1 / np.sqrt(2)
    return m


_FB = mel_filterbank()
_DCT = _dct_matrix(13, N_MELS)


def mfcc_from_power(P: np.ndarray, noise_psd: np.ndarray | None = None) -> np.ndarray:
    """(T, 13) MFCCs; optional spectral subtraction first."""
    if noise_psd is not None:
        P = spectral_subtract(P, noise_psd)
    logmel = np.log(P @ _FB.T + 1e-8)
    return logmel @ _DCT.T


def cmvn(F: np.ndarray) -> np.ndarray:
    """Per-utterance mean/variance normalisation (removes gain and channel colouring)."""
    return (F - F.mean(0, keepdims=True)) / (F.std(0, keepdims=True) + 1e-6)


def time_normalize(F: np.ndarray, n: int = 20) -> np.ndarray:
    """Linearly resample the time axis to ``n`` frames."""
    T = len(F)
    if T == 0:
        raise ValueError("empty feature sequence")
    if T == 1:
        return np.repeat(F, n, axis=0)
    src = np.linspace(0, T - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None]
    return F[lo] * (1 - w) + F[hi] * w


def utterance_vector(P: np.ndarray, noise_psd: np.ndarray | None = None, n: int = 20) -> np.ndarray:
    """Fixed-length vector for a segment: normalised MFCC trajectory + log duration."""
    F = cmvn(mfcc_from_power(P, noise_psd))
    return np.concatenate([time_normalize(F, n).ravel(), [np.log(len(P) * HOP_MS / 1000.0)]])


def sequence_features(P: np.ndarray, noise_psd: np.ndarray | None = None, step: int = 2) -> np.ndarray:
    """Variable-length MFCC sequence (CMVN, decimated) for DTW."""
    return cmvn(mfcc_from_power(P, noise_psd))[::step]
